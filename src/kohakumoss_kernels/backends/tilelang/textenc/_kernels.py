"""Text-encoder kernels in TileLang (the sm70 / sm75 path): token-serial gated delta
rule and causal GQA varlen attention forward. Token and segment counts are symbolic, so
one compile serves every batch (compiled with `_attention_kernels.compile_kernel`). See
docs/internals/text-encoder.md.
"""

import tilelang.language as T

ACC = "float32"


def delta_recurrent(hk: int, hv: int, dk: int, dv: int, bv: int, dtype: str, threads: int):  # fmt: skip
    """(Q, K, V, A, B, ALOG, DTB, CU, Out, scale); grid (segments, value heads, value
    tiles), one token per step, fp32 state; o_t = e^g S^T q + delta (k . q)."""
    n = T.dynamic("n")
    b1 = T.dynamic("b1")
    groups = hv // hk

    @T.prim_func
    def kernel(
        Q: T.Tensor((n, hk, dk), dtype),
        K: T.Tensor((n, hk, dk), dtype),
        V: T.Tensor((n, hv, dv), dtype),
        A: T.Tensor((n, hv), dtype),
        B: T.Tensor((n, hv), dtype),
        ALOG: T.Tensor((hv,), ACC),
        DTB: T.Tensor((hv,), ACC),
        CU: T.Tensor((b1,), "int32"),
        Out: T.Tensor((n, hv, dv), dtype),
        scale: T.float32,
    ):
        with T.Kernel(b1 - 1, hv, dv // bv, threads=threads) as (seg, head, vt):
            state = T.alloc_fragment((dk, bv), ACC)
            prod = T.alloc_fragment((dk, bv), ACC)
            sk = T.alloc_fragment((bv,), ACC)
            sq = T.alloc_fragment((bv,), ACC)
            vk = T.alloc_fragment((dk,), ACC)
            vq = T.alloc_fragment((dk,), ACC)
            vx = T.alloc_fragment((dk,), ACC)
            nk = T.alloc_fragment((1,), ACC)
            nq = T.alloc_fragment((1,), ACC)
            kq = T.alloc_fragment((1,), ACC)
            k_s = T.alloc_shared((dk,), ACC)
            q_s = T.alloc_shared((dk,), ACC)
            v_s = T.alloc_shared((bv,), ACC)
            d_s = T.alloc_shared((bv,), ACC)
            sc = T.alloc_shared((5,), ACC)
            kh = head // groups
            T.clear(state)
            for t in T.serial(CU[seg], CU[seg + 1]):
                for i in T.Parallel(dk):
                    k_s[i] = T.Cast(ACC, K[t, kh, i])
                    q_s[i] = T.Cast(ACC, Q[t, kh, i])
                for j in T.Parallel(bv):
                    v_s[j] = T.Cast(ACC, V[t, head, vt * bv + j])
                for i in T.Parallel(dk):
                    vk[i] = k_s[i] * k_s[i]
                    vq[i] = q_s[i] * q_s[i]
                    vx[i] = k_s[i] * q_s[i]
                T.reduce_sum(vk, nk, dim=0)
                T.reduce_sum(vq, nq, dim=0)
                T.reduce_sum(vx, kq, dim=0)
                for i, j in T.Parallel(dk, bv):
                    prod[i, j] = state[i, j] * k_s[i]
                T.reduce_sum(prod, sk, dim=0)
                for i, j in T.Parallel(dk, bv):
                    prod[i, j] = state[i, j] * q_s[i]
                T.reduce_sum(prod, sq, dim=0)
                x = T.Cast(ACC, A[t, head]) + DTB[head]
                sp = T.if_then_else(x > 20.0, x, T.log1p(T.exp(x)))
                sc[0] = T.exp(-T.exp(ALOG[head]) * sp)
                sc[1] = T.sigmoid(T.Cast(ACC, B[t, head]))
                sc[2] = 1.0 / T.sqrt(nk[0] + 1e-6)
                sc[3] = scale / T.sqrt(nq[0] + 1e-6)
                sc[4] = kq[0]
                for j in T.Parallel(bv):
                    d_s[j] = (v_s[j] - sc[0] * sc[2] * sk[j]) * sc[1]
                for j in T.Parallel(bv):
                    Out[t, head, vt * bv + j] = sc[0] * sc[3] * sq[j] + d_s[j] * (
                        sc[4] * sc[2] * sc[3]
                    )
                for i, j in T.Parallel(dk, bv):
                    state[i, j] = state[i, j] * sc[0] + k_s[i] * sc[2] * d_s[j]

    return kernel


def attn_causal(h: int, hkv: int, d: int, dtype: str, bm: int, bn: int, threads: int):
    """(Q, K, V, CU, Out, mb, c) causal GQA forward; grid (query blocks of the longest
    segment, query heads, segments); key blocks past the diagonal are skipped."""
    n = T.dynamic("n")
    b1 = T.dynamic("b1")
    groups = h // hkv

    @T.prim_func
    def kernel(
        Q: T.Tensor((n, h, d), dtype),
        K: T.Tensor((n, hkv, d), dtype),
        V: T.Tensor((n, hkv, d), dtype),
        CU: T.Tensor((b1,), "int32"),
        Out: T.Tensor((n, h, d), dtype),
        mb: T.int32,
        c: T.float32,
    ):
        with T.Kernel(mb, h, b1 - 1, threads=threads) as (qb, head, seg):
            q_s = T.alloc_shared((bm, d), dtype)
            k_s = T.alloc_shared((bn, d), dtype)
            v_s = T.alloc_shared((bn, d), dtype)
            s = T.alloc_fragment((bm, bn), ACC)
            p_s = T.alloc_shared((bm, bn), dtype)
            acc = T.alloc_fragment((bm, d), ACC)
            m_i = T.alloc_fragment((bm,), ACC)
            m_prev = T.alloc_fragment((bm,), ACC)
            alpha = T.alloc_fragment((bm,), ACC)
            row_sum = T.alloc_fragment((bm,), ACC)
            l_i = T.alloc_fragment((bm,), ACC)
            kvh = head // groups
            start = CU[seg]
            seqlen = CU[seg + 1] - start
            if qb * bm < seqlen:
                q0 = start + qb * bm
                T.copy(Q[q0 : q0 + bm, head, :], q_s)
                T.fill(acc, 0)
                T.fill(l_i, 0)
                T.fill(m_i, -T.infinity(ACC))
                hi = T.min(qb * bm + bm, seqlen)
                for kb in T.serial(T.ceildiv(hi, bn)):
                    k0 = start + kb * bn
                    T.copy(K[k0 : k0 + bn, kvh, :], k_s)
                    T.clear(s)
                    T.gemm(
                        q_s, k_s, s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    for i, j in T.Parallel(bm, bn):
                        s[i, j] = T.if_then_else(
                            (kb * bn + j <= qb * bm + i) & (kb * bn + j < seqlen),
                            s[i, j],
                            -T.infinity(ACC),
                        )
                    T.copy(m_i, m_prev)
                    T.reduce_max(s, m_i, dim=1, clear=False)
                    for i in T.Parallel(bm):
                        alpha[i] = T.exp2((m_prev[i] - m_i[i]) * c)
                    for i, j in T.Parallel(bm, bn):
                        s[i, j] = T.exp2((s[i, j] - m_i[i]) * c)
                    T.reduce_sum(s, row_sum, dim=1)
                    for i in T.Parallel(bm):
                        l_i[i] = l_i[i] * alpha[i] + row_sum[i]
                    for i, j in T.Parallel(bm, d):
                        acc[i, j] *= alpha[i]
                    T.copy(s, p_s)
                    T.copy(V[k0 : k0 + bn, kvh, :], v_s)
                    T.gemm(p_s, v_s, acc, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(bm, d):
                    if qb * bm + i < seqlen:
                        Out[start + qb * bm + i, head, j] = acc[i, j] / l_i[i]

    return kernel
