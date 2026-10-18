"""Varlen flash attention in TileLang (Volta / Turing MMA): fwd, delta, dK/dV, dQ.

Same semantics as the Triton kernels: packed `[N, H, D]`, segments from `CU`, the grid's
first axis covers the longest segment (`mb` blocks) and programs past their segment's
end do nothing. Row statistics are two fp32 planes `M` (raw score max) and `L`
(log2 row sum); `c = scale * log2(e)`. Token count and segment count are symbolic, so
one compile serves every batch. See docs/internals/kernel-dev.md.
"""

import tilelang
import tilelang.language as T

ACC = "float32"
DELTA_BLOCK = 64


def _pass_configs(safe: bool) -> dict:
    out = {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}
    if not safe:
        out["tl.disable_safe_memory_legalize"] = True
    return out


def compile_kernel(prim, arch: str, safe: bool = True):
    """Compile for `arch`; `safe=False` drops TileLang's automatic bounds guards (tests)."""
    return tilelang.compile(
        prim,
        target={"kind": "cuda", "arch": arch},
        pass_configs=_pass_configs(safe),
    )


def fwd(
    h: int,
    d: int,
    dtype: str,
    bm: int,
    bn: int,
    threads: int,
    qreg: bool = False,
    share: bool = False,
):
    """(Q, K, V, CU, Out, M, L, mb, c) -> fills Out, M, L. `qreg` holds the query tile
    in registers, `share` streams K and V through one shared buffer (less shared
    memory, more resident programs per SM)."""
    n = T.dynamic("n")
    b1 = T.dynamic("b1")

    @T.prim_func
    def kernel(
        Q: T.Tensor((n, h, d), dtype),
        K: T.Tensor((n, h, d), dtype),
        V: T.Tensor((n, h, d), dtype),
        CU: T.Tensor((b1,), "int32"),
        Out: T.Tensor((n, h, d), dtype),
        M: T.Tensor((h, n), ACC),
        L: T.Tensor((h, n), ACC),
        mb: T.int32,
        c: T.float32,
    ):
        with T.Kernel(mb, h, b1 - 1, threads=threads) as (qb, head, seg):
            q_s = (
                T.alloc_fragment((bm, d), dtype)
                if qreg
                else T.alloc_shared((bm, d), dtype)
            )
            k_s = T.alloc_shared((bn, d), dtype)
            v_s = k_s if share else T.alloc_shared((bn, d), dtype)
            s = T.alloc_fragment((bm, bn), ACC)
            p_s = T.alloc_shared((bm, bn), dtype)
            acc = T.alloc_fragment((bm, d), ACC)
            m_i = T.alloc_fragment((bm,), ACC)
            m_prev = T.alloc_fragment((bm,), ACC)
            alpha = T.alloc_fragment((bm,), ACC)
            row_sum = T.alloc_fragment((bm,), ACC)
            l_i = T.alloc_fragment((bm,), ACC)
            start = CU[seg]
            seqlen = CU[seg + 1] - start
            if qb * bm < seqlen:
                q0 = start + qb * bm
                T.copy(Q[q0 : q0 + bm, head, :], q_s)
                T.fill(acc, 0)
                T.fill(l_i, 0)
                T.fill(m_i, -T.infinity(ACC))
                for kb in T.serial(T.ceildiv(seqlen, bn)):
                    k0 = start + kb * bn
                    T.copy(K[k0 : k0 + bn, head, :], k_s)
                    T.clear(s)
                    T.gemm(
                        q_s, k_s, s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    for i, j in T.Parallel(bm, bn):
                        s[i, j] = T.if_then_else(
                            kb * bn + j < seqlen, s[i, j], -T.infinity(ACC)
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
                    T.copy(V[k0 : k0 + bn, head, :], v_s)
                    T.gemm(p_s, v_s, acc, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(bm, d):
                    if qb * bm + i < seqlen:
                        Out[start + qb * bm + i, head, j] = acc[i, j] / l_i[i]
                for i in T.Parallel(bm):
                    if qb * bm + i < seqlen:
                        M[head, start + qb * bm + i] = m_i[i]
                        L[head, start + qb * bm + i] = T.log2(l_i[i])

    return kernel


def delta(h: int, d: int, dtype: str):
    """(O, dO, DELTA) -> DELTA[h, n] = sum_d O * dO (fp32)."""
    n = T.dynamic("n")
    blk = DELTA_BLOCK

    @T.prim_func
    def kernel(
        Out: T.Tensor((n, h, d), dtype),
        dO: T.Tensor((n, h, d), dtype),
        Delta: T.Tensor((h, n), ACC),
    ):
        with T.Kernel(T.ceildiv(n, blk), h, threads=128) as (tb, head):
            o = T.alloc_fragment((blk, d), dtype)
            do = T.alloc_fragment((blk, d), dtype)
            prod = T.alloc_fragment((blk, d), ACC)
            out = T.alloc_fragment((blk,), ACC)
            T.copy(Out[tb * blk : tb * blk + blk, head, :], o)
            T.copy(dO[tb * blk : tb * blk + blk, head, :], do)
            for i, j in T.Parallel(blk, d):
                prod[i, j] = T.Cast(ACC, o[i, j]) * T.Cast(ACC, do[i, j])
            T.reduce_sum(prod, out, dim=1)
            for i in T.Parallel(blk):
                if tb * blk + i < n:
                    Delta[head, tb * blk + i] = out[i]

    return kernel


def dkdv(
    h: int,
    d: int,
    dtype: str,
    bm: int,
    bn: int,
    threads: int,
    kvreg: bool = False,
    pds: bool = False,
):
    """(Q, K, V, dO, M, L, Delta, CU, dK, dV, mb, scale, c); one program per key block.
    `kvreg` keeps the key / value tile in registers, `pds` stages P and dS through one
    shared buffer."""
    n = T.dynamic("n")
    b1 = T.dynamic("b1")

    @T.prim_func
    def kernel(
        Q: T.Tensor((n, h, d), dtype),
        K: T.Tensor((n, h, d), dtype),
        V: T.Tensor((n, h, d), dtype),
        dO: T.Tensor((n, h, d), dtype),
        M: T.Tensor((h, n), ACC),
        L: T.Tensor((h, n), ACC),
        Delta: T.Tensor((h, n), ACC),
        CU: T.Tensor((b1,), "int32"),
        dK: T.Tensor((n, h, d), dtype),
        dV: T.Tensor((n, h, d), dtype),
        mb: T.int32,
        scale: T.float32,
        c: T.float32,
    ):
        with T.Kernel(mb, h, b1 - 1, threads=threads) as (kb, head, seg):
            k_s = (
                T.alloc_fragment((bn, d), dtype)
                if kvreg
                else T.alloc_shared((bn, d), dtype)
            )
            v_s = (
                T.alloc_fragment((bn, d), dtype)
                if kvreg
                else T.alloc_shared((bn, d), dtype)
            )
            q_s = T.alloc_shared((bm, d), dtype)
            do_s = T.alloc_shared((bm, d), dtype)
            m_s = T.alloc_shared((bm,), ACC)
            l_s = T.alloc_shared((bm,), ACC)
            d_s = T.alloc_shared((bm,), ACC)
            p_t = T.alloc_fragment((bn, bm), ACC)
            dp_t = T.alloc_fragment((bn, bm), ACC)
            p_s = T.alloc_shared((bn, bm), dtype)
            ds_s = p_s if pds else T.alloc_shared((bn, bm), dtype)
            dk = T.alloc_fragment((bn, d), ACC)
            dv = T.alloc_fragment((bn, d), ACC)
            start = CU[seg]
            seqlen = CU[seg + 1] - start
            if kb * bn < seqlen:
                k0 = start + kb * bn
                T.copy(K[k0 : k0 + bn, head, :], k_s)
                T.copy(V[k0 : k0 + bn, head, :], v_s)
                T.clear(dk)
                T.clear(dv)
                for qb in T.serial(T.ceildiv(seqlen, bm)):
                    q0 = start + qb * bm
                    T.copy(Q[q0 : q0 + bm, head, :], q_s)
                    T.copy(dO[q0 : q0 + bm, head, :], do_s)
                    for j in T.Parallel(bm):
                        row = qb * bm + j < seqlen
                        m_s[j] = T.if_then_else(row, M[head, q0 + j], 0)
                        l_s[j] = T.if_then_else(row, L[head, q0 + j], 0)
                        d_s[j] = T.if_then_else(row, Delta[head, q0 + j], 0)
                    T.clear(p_t)
                    T.gemm(
                        k_s, q_s, p_t, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    for i, j in T.Parallel(bn, bm):
                        p_t[i, j] = T.if_then_else(
                            qb * bm + j < seqlen,
                            T.exp2((p_t[i, j] - m_s[j]) * c - l_s[j]),
                            0,
                        )
                    T.clear(dp_t)
                    T.gemm(
                        v_s,
                        do_s,
                        dp_t,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(p_t, p_s)
                    T.gemm(p_s, do_s, dv, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(bn, bm):
                        dp_t[i, j] = p_t[i, j] * (dp_t[i, j] - d_s[j])
                    T.copy(dp_t, ds_s)
                    T.gemm(ds_s, q_s, dk, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(bn, d):
                    if kb * bn + i < seqlen:
                        dK[start + kb * bn + i, head, j] = dk[i, j] * scale
                        dV[start + kb * bn + i, head, j] = dv[i, j]

    return kernel


def dq(
    h: int,
    d: int,
    dtype: str,
    bm: int,
    bn: int,
    threads: int,
    qreg: bool = False,
    share: bool = False,
):
    """(Q, K, V, dO, M, L, Delta, CU, dQ, mb, scale, c); one program per query block.
    `qreg` keeps the query / dO tile in registers, `share` streams K and V through one
    shared buffer."""
    n = T.dynamic("n")
    b1 = T.dynamic("b1")

    @T.prim_func
    def kernel(
        Q: T.Tensor((n, h, d), dtype),
        K: T.Tensor((n, h, d), dtype),
        V: T.Tensor((n, h, d), dtype),
        dO: T.Tensor((n, h, d), dtype),
        M: T.Tensor((h, n), ACC),
        L: T.Tensor((h, n), ACC),
        Delta: T.Tensor((h, n), ACC),
        CU: T.Tensor((b1,), "int32"),
        dQ: T.Tensor((n, h, d), dtype),
        mb: T.int32,
        scale: T.float32,
        c: T.float32,
    ):
        with T.Kernel(mb, h, b1 - 1, threads=threads) as (qb, head, seg):
            q_s = (
                T.alloc_fragment((bm, d), dtype)
                if qreg
                else T.alloc_shared((bm, d), dtype)
            )
            do_s = (
                T.alloc_fragment((bm, d), dtype)
                if qreg
                else T.alloc_shared((bm, d), dtype)
            )
            k_s = T.alloc_shared((bn, d), dtype)
            v_s = k_s if share else T.alloc_shared((bn, d), dtype)
            m_f = T.alloc_fragment((bm,), ACC)
            l_f = T.alloc_fragment((bm,), ACC)
            d_f = T.alloc_fragment((bm,), ACC)
            p = T.alloc_fragment((bm, bn), ACC)
            dp = T.alloc_fragment((bm, bn), ACC)
            ds_s = T.alloc_shared((bm, bn), dtype)
            acc = T.alloc_fragment((bm, d), ACC)
            start = CU[seg]
            seqlen = CU[seg + 1] - start
            if qb * bm < seqlen:
                q0 = start + qb * bm
                T.copy(Q[q0 : q0 + bm, head, :], q_s)
                T.copy(dO[q0 : q0 + bm, head, :], do_s)
                for i in T.Parallel(bm):
                    row = qb * bm + i < seqlen
                    m_f[i] = T.if_then_else(row, M[head, q0 + i], 0)
                    l_f[i] = T.if_then_else(row, L[head, q0 + i], 0)
                    d_f[i] = T.if_then_else(row, Delta[head, q0 + i], 0)
                T.clear(acc)
                for kb in T.serial(T.ceildiv(seqlen, bn)):
                    k0 = start + kb * bn
                    T.copy(V[k0 : k0 + bn, head, :], v_s)
                    T.clear(dp)
                    T.gemm(
                        do_s, v_s, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    T.copy(K[k0 : k0 + bn, head, :], k_s)
                    T.clear(p)
                    T.gemm(
                        q_s, k_s, p, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )
                    for i, j in T.Parallel(bm, bn):
                        p[i, j] = T.if_then_else(
                            kb * bn + j < seqlen,
                            T.exp2((p[i, j] - m_f[i]) * c - l_f[i]),
                            0,
                        )
                    for i, j in T.Parallel(bm, bn):
                        dp[i, j] = p[i, j] * (dp[i, j] - d_f[i])
                    T.copy(dp, ds_s)
                    T.gemm(ds_s, k_s, acc, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(bm, d):
                    if qb * bm + i < seqlen:
                        dQ[start + qb * bm + i, head, j] = acc[i, j] * scale

    return kernel
