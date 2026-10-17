"""TileLang RMSNorm backward prims of the `df` reduce mode: double-float tile partials
and the parameter-gradient epilogue. See docs/internals/fp16.md."""

import tilelang.language as T

from kohakumoss_kernels.backends.tilelang._df import (
    df_add,
    df_div,
    df_mul,
    df_scale,
    extract_bits,
    rstd_into,
    split_hi,
    split_sigma,
    two_sum,
)

F32 = "float32"
I32 = "int32"


def bwd_df_prim(d, bm, threads, dtype):
    """The norm backward with double-float partials: exact g * x (16-bit inputs), rstd
    recomputed in pairs from an exact hi/lo split sum of x * x."""
    n = T.dynamic("n")
    lc = T.dynamic("lc")
    tt = T.dynamic("t")
    k = extract_bits(d)

    @T.prim_func
    def main(
        g: T.Tensor((n, d), dtype),
        x: T.Tensor((n, d), dtype),
        rstd: T.Tensor((n,), F32),
        w: T.Tensor((d,), F32),
        scale: T.Tensor((lc,), F32),
        tseg: T.Tensor((tt,), I32),
        tstart: T.Tensor((tt,), I32),
        tstop: T.Tensor((tt,), I32),
        dx: T.Tensor((n, d), dtype),
        part: T.Tensor((tt, bm, 2, 2, d), F32),
        cr: T.int32,
        cc: T.int32,
        eps: T.float32,
    ):
        with T.Kernel(tt, threads=threads) as t:
            acc_gx = T.alloc_fragment((bm, d), F32)
            c_gx = T.alloc_fragment((bm, d), F32)
            acc_g = T.alloc_fragment((bm, d), F32)
            c_g = T.alloc_fragment((bm, d), F32)
            sq = T.alloc_fragment((bm, d), F32)
            sl = T.alloc_fragment((bm, d), F32)
            prod = T.alloc_fragment((bm, d), F32)
            c = T.alloc_fragment((bm,), F32)
            mx = T.alloc_fragment((bm,), F32)
            shs = T.alloc_fragment((bm,), F32)
            sls = T.alloc_fragment((bm,), F32)
            rs = T.alloc_fragment((bm,), F32)
            rh = T.alloc_fragment((bm,), F32)
            rl = T.alloc_fragment((bm,), F32)
            t0 = T.alloc_fragment((bm,), F32)
            t1 = T.alloc_fragment((bm,), F32)
            t2 = T.alloc_fragment((bm,), F32)
            t3 = T.alloc_fragment((bm,), F32)
            t4 = T.alloc_fragment((bm,), F32)
            sb = tseg[t]
            start = tstart[t]
            stop = tstop[t]
            T.clear(acc_gx)
            T.clear(c_gx)
            T.clear(acc_g)
            T.clear(c_g)
            for it in T.serial(T.ceildiv(stop - start, bm)):
                for i in T.Parallel(bm):
                    r = start + it * bm + i
                    rs[i] = T.if_then_else(r < stop, rstd[T.min(r, n - 1)], 0.0)
                for i, j in T.Parallel(bm, d):
                    r = start + it * bm + i
                    ok = r < stop
                    rr = T.min(r, n - 1)
                    gv = T.if_then_else(ok, T.cast(g[rr, j], F32), 0.0)
                    x32 = T.if_then_else(ok, T.cast(x[rr, j], F32), 0.0)
                    mj = w[j] * (1.0 + scale[sb * cr + j * cc])
                    prod[i, j] = gv * mj * (x32 * rs[i])
                    sq[i, j] = T.ieee_mul(x32, x32)
                T.reduce_sum(prod, c, dim=1)
                T.reduce_max(sq, mx, dim=1)
                for i, j in T.Parallel(bm, d):
                    hi = split_hi(sq[i, j], split_sigma(mx[i], k))
                    sl[i, j] = T.ieee_sub(sq[i, j], hi)
                    sq[i, j] = hi
                T.reduce_sum(sq, shs, dim=1)
                T.reduce_sum(sl, sls, dim=1)
                for i in T.Parallel(bm):
                    sh, sl2 = two_sum(shs[i], sls[i])
                    t0[i] = sh
                    t1[i] = sl2
                for i in T.Parallel(bm):
                    qh, ql = df_div(t0[i], t1[i], T.float32(d))
                    t2[i] = qh
                    t3[i] = ql
                for i in T.Parallel(bm):
                    rstd_into(rh, rl, t0, t1, t4, i, t2[i], t3[i], eps)
                for i in T.Parallel(bm):
                    ok = start + it * bm + i < stop
                    rh[i] = T.if_then_else(ok, rh[i], 0.0)
                    rl[i] = T.if_then_else(ok, rl[i], 0.0)
                for i, j in T.Parallel(bm, d):
                    r = start + it * bm + i
                    rr = T.min(r, n - 1)
                    gv = T.if_then_else(r < stop, T.cast(g[rr, j], F32), 0.0)
                    x32 = T.if_then_else(r < stop, T.cast(x[rr, j], F32), 0.0)
                    ph, pl = df_scale(T.ieee_mul(gv, x32), rh[i], rl[i])
                    sq[i, j] = ph
                    sl[i, j] = pl
                for i, j in T.Parallel(bm, d):
                    r = start + it * bm + i
                    ok = r < stop
                    rr = T.min(r, n - 1)
                    gv = T.if_then_else(ok, T.cast(g[rr, j], F32), 0.0)
                    x32 = T.if_then_else(ok, T.cast(x[rr, j], F32), 0.0)
                    ah, al = df_add(acc_gx[i, j], c_gx[i, j], sq[i, j], sl[i, j])
                    acc_gx[i, j] = ah
                    c_gx[i, j] = al
                    bh, bl = df_add(acc_g[i, j], c_g[i, j], gv, T.float32(0))
                    acc_g[i, j] = bh
                    c_g[i, j] = bl
                    mj = w[j] * (1.0 + scale[sb * cr + j * cc])
                    if ok:
                        dx[r, j] = T.cast(
                            rs[i] * (gv * mj - x32 * rs[i] * c[i] / d), dtype
                        )
            for i, j in T.Parallel(bm, d):
                part[t, i, 0, 0, j] = acc_gx[i, j]
                part[t, i, 0, 1, j] = c_gx[i, j]
                part[t, i, 1, 0, j] = acc_g[i, j]
                part[t, i, 1, 1, j] = c_g[i, j]

    return main


def grads_prim(d, bd, threads):
    """d_weight [d], d_shift [b, d], d_scale [b, d] (fp32, rounded once) from the pair
    sums [2 (S_gx, S_g), 2 (hi, lo), b, d] in double-float."""
    b = T.dynamic("b")
    lc = T.dynamic("lc")

    @T.prim_func
    def main(
        sums: T.Tensor((2, 2, b, d), F32),
        w: T.Tensor((d,), F32),
        scale: T.Tensor((lc,), F32),
        dw: T.Tensor((d,), F32),
        dshift: T.Tensor((b, d), F32),
        dscale: T.Tensor((b, d), F32),
        cr: T.int32,
        cc: T.int32,
    ):
        with T.Kernel(T.ceildiv(d, bd), threads=threads) as cb:
            ah = T.alloc_fragment((bd,), F32)
            al = T.alloc_fragment((bd,), F32)
            th = T.alloc_fragment((bd,), F32)
            tl = T.alloc_fragment((bd,), F32)
            T.clear(ah)
            T.clear(al)
            for sb in T.serial(b):
                for j in T.Parallel(bd):
                    col = T.min(cb * bd + j, d - 1)
                    oh, ol = two_sum(T.float32(1), scale[sb * cr + col * cc])
                    th[j] = oh
                    tl[j] = ol
                for j in T.Parallel(bd):
                    col = T.min(cb * bd + j, d - 1)
                    gxh = sums[0, 0, sb, col]
                    ph, pl = df_mul(gxh, sums[0, 1, sb, col], th[j], tl[j])
                    th[j] = ph
                    tl[j] = pl
                for j in T.Parallel(bd):
                    col = T.min(cb * bd + j, d - 1)
                    h, lw = df_add(ah[j], al[j], th[j], tl[j])
                    ah[j] = h
                    al[j] = lw
                    gxh = sums[0, 0, sb, col]
                    gxl = sums[0, 1, sb, col]
                    sh, sc = df_scale(w[col], gxh, gxl)
                    if cb * bd + j < d:
                        dshift[sb, col] = T.ieee_add(
                            sums[1, 0, sb, col], sums[1, 1, sb, col]
                        )
                        dscale[sb, col] = T.ieee_add(sh, sc)
            for j in T.Parallel(bd):
                if cb * bd + j < d:
                    dw[cb * bd + j] = T.ieee_add(ah[j], al[j])

    return main
