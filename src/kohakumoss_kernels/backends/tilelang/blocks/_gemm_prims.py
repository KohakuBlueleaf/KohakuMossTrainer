"""TileLang GEMMs `A [M, K] @ W[rows, K]^T` with MOSS epilogues (fp16 in, fp32 accumulate).

Tensor-core MMA through `T.gemm` on sm70+; the token count is dynamic. The epilogue
reads the fp32 accumulator and rounds once at the store.
"""

import tilelang.language as T

F16 = "float16"
F32 = "float32"
I32 = "int32"


def situ_gemm_prim(k, f, bm, bn, bk, threads, stages, save):
    """out [M, F] = situ(A W_up^T, A W_gate^T); ug [M, 2F] stored when `save` (any
    placeholder tensor otherwise)."""
    m = T.dynamic("m")
    ug_shape = (m, 2 * f) if save else (T.dynamic("mu"), T.dynamic("cu"))

    @T.prim_func
    def main(
        a: T.Tensor((m, k), F16),
        w: T.Tensor((2 * f, k), F16),
        out: T.Tensor((m, f), F16),
        ug: T.Tensor(ug_shape, F16),
    ):
        with T.Kernel(T.ceildiv(f, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
            a_s = T.alloc_shared((bm, bk), F16)
            wu_s = T.alloc_shared((bn, bk), F16)
            wg_s = T.alloc_shared((bn, bk), F16)
            acc_u = T.alloc_fragment((bm, bn), F32)
            acc_g = T.alloc_fragment((bm, bn), F32)
            T.clear(acc_u)
            T.clear(acc_g)
            for kk in T.Pipelined(T.ceildiv(k, bk), num_stages=stages):
                T.copy(a[by * bm, kk * bk], a_s)
                T.copy(w[bx * bn, kk * bk], wu_s)
                T.copy(w[f + bx * bn, kk * bk], wg_s)
                T.gemm(a_s, wu_s, acc_u, transpose_B=True)
                T.gemm(a_s, wg_s, acc_g, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                row = by * bm + i
                col = bx * bn + j
                u = acc_u[i, j]
                g = acc_g[i, j]
                v = (4.0 * T.tanh(g * 0.25) * T.sigmoid(g)) * (25.0 * T.tanh(u * 0.04))
                if row < m and col < f:
                    out[row, col] = T.cast(v, F16)
                    if save:
                        ug[row, col] = T.cast(u, F16)
                        ug[row, f + col] = T.cast(g, F16)

    return main


def residual_gemm_prim(k, n, bm, bn, bk, threads, stages, gated, scaled, save):
    """out = x * r + (A W^T) * gate[seg] (gate flat with row / column strides gr, gc);
    o = A W^T stored when `save`."""
    m = T.dynamic("m")
    lg = T.dynamic("lg")

    @T.prim_func
    def main(
        a: T.Tensor((m, k), F16),
        w: T.Tensor((n, k), F16),
        x: T.Tensor((m, n), F16),
        gate: T.Tensor((lg,), F32),
        seg: T.Tensor((m,), I32),
        out: T.Tensor((m, n), F16),
        o: T.Tensor((m, n), F16),
        r: T.float32,
        gr: T.int32,
        gc: T.int32,
    ):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
            a_s = T.alloc_shared((bm, bk), F16)
            w_s = T.alloc_shared((bn, bk), F16)
            acc = T.alloc_fragment((bm, bn), F32)
            T.clear(acc)
            for kk in T.Pipelined(T.ceildiv(k, bk), num_stages=stages):
                T.copy(a[by * bm, kk * bk], a_s)
                T.copy(w[bx * bn, kk * bk], w_s)
                T.gemm(a_s, w_s, acc, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                row = by * bm + i
                col = bx * bn + j
                rr = T.min(row, m - 1)
                cc = T.min(col, n - 1)
                gv = T.if_then_else(gated, gate[seg[rr] * gr + cc * gc], 1.0)
                xv = T.cast(x[rr, cc], F32)
                xv = T.if_then_else(scaled, xv * r, xv)
                if row < m and col < n:
                    out[row, col] = T.cast(xv + acc[i, j] * gv, F16)
                    if save:
                        o[row, col] = T.cast(acc[i, j], F16)

    return main


def qk_gemm_prim(k, heads, half, nb, axes, bm, bk, threads, stages, save):
    """One program = bm tokens x one head of q (bx < heads) or k: GEMM halves, per-head
    RMSNorm with weight, split-half RoPE from positions."""
    m = T.dynamic("m")
    d = heads * 2 * half

    @T.prim_func
    def main(
        a: T.Tensor((m, k), F16),
        wq: T.Tensor((d, k), F16),
        wk: T.Tensor((d, k), F16),
        qw: T.Tensor((2 * half,), F32),
        kw: T.Tensor((2 * half,), F32),
        pos: T.Tensor((m, axes), F32),
        bands: T.Tensor((nb,), F32),
        qo: T.Tensor((m, d), F16),
        ko: T.Tensor((m, d), F16),
        qp: T.Tensor((m, d), F16),
        kp: T.Tensor((m, d), F16),
        qr: T.Tensor((m, heads), F32),
        kr: T.Tensor((m, heads), F32),
        eps: T.float32,
    ):
        with T.Kernel(2 * heads, T.ceildiv(m, bm), threads=threads) as (bx, by):
            a_s = T.alloc_shared((bm, bk), F16)
            w1_s = T.alloc_shared((half, bk), F16)
            w2_s = T.alloc_shared((half, bk), F16)
            u = T.alloc_fragment((bm, half), F32)
            v = T.alloc_fragment((bm, half), F32)
            sq = T.alloc_fragment((bm, half), F32)
            ss = T.alloc_fragment((bm,), F32)
            is_k = bx >= heads
            head = T.if_then_else(is_k, bx - heads, bx)
            T.clear(u)
            T.clear(v)
            for kk in T.Pipelined(T.ceildiv(k, bk), num_stages=stages):
                T.copy(a[by * bm, kk * bk], a_s)
                for i, j in T.Parallel(half, bk):
                    w1_s[i, j] = T.if_then_else(
                        is_k,
                        wk[head * 2 * half + i, kk * bk + j],
                        wq[head * 2 * half + i, kk * bk + j],
                    )
                    w2_s[i, j] = T.if_then_else(
                        is_k,
                        wk[head * 2 * half + half + i, kk * bk + j],
                        wq[head * 2 * half + half + i, kk * bk + j],
                    )
                T.gemm(a_s, w1_s, u, transpose_B=True)
                T.gemm(a_s, w2_s, v, transpose_B=True)
            for i, j in T.Parallel(bm, half):
                sq[i, j] = u[i, j] * u[i, j] + v[i, j] * v[i, j]
            T.reduce_sum(sq, ss, dim=1)
            for i in T.Parallel(bm):
                ss[i] = 1.0 / T.sqrt(ss[i] / (2 * half) + eps)
            for i, j in T.Parallel(bm, half):
                row = by * bm + i
                rr = T.min(row, m - 1)
                col = head * 2 * half + j
                nw1 = T.if_then_else(is_k, kw[j], qw[j])
                nw2 = T.if_then_else(is_k, kw[j + half], qw[j + half])
                an = u[i, j] * ss[i] * nw1
                bn_ = v[i, j] * ss[i] * nw2
                ph = pos[rr, j // nb] * bands[j % nb]
                s = T.sin(ph)
                c = T.cos(ph)
                oa = T.cast(an * c - bn_ * s, F16)
                ob = T.cast(bn_ * c + an * s, F16)
                if row < m:
                    if is_k:
                        ko[row, col] = oa
                        ko[row, col + half] = ob
                        if save:
                            kp[row, col] = T.cast(u[i, j], F16)
                            kp[row, col + half] = T.cast(v[i, j], F16)
                            if j == 0:
                                kr[row, head] = ss[i]
                    else:
                        qo[row, col] = oa
                        qo[row, col + half] = ob
                        if save:
                            qp[row, col] = T.cast(u[i, j], F16)
                            qp[row, col + half] = T.cast(v[i, j], F16)
                            if j == 0:
                                qr[row, head] = ss[i]

    return main
