"""TileLang RMSNorm and RMSNorm + per-sample modulation, forward and backward.

Same algorithm as the Triton backend: forward holds `[BM, D]` rows in a fragment and
keeps rstd; backward runs one program per segment tile, writes dx and per-tile fp32
column partials of sum(g * xhat) and sum(g); segment sums and an epilogue give the
parameter gradients. Small parameters (weight, shift, scale) are passed as fp32. See
docs/internals/fp16.md.
"""

import tilelang.language as T
import torch
from torch import Tensor
from torch.library import custom_op

from kohakumoss_kernels.backends.compile.choice import kernel_or_compile, shape_key
from kohakumoss_kernels.backends.eager import norm as eager_norm
from kohakumoss_kernels.backends.tilelang.blocks._blockops import (
    DTYPE,
    F32,
    FP16_SUPPORTS,
    I32,
    CompileError,
    compiled,
    f32,
    per_sample,
    segment_ids,
    segment_sums,
    tile_plan,
    tuned_run,
)
from kohakumoss_kernels.registry import register

__all__ = ["rms_norm", "rms_norm_mod"]


def _fwd_prim(d, bm, threads, dtype, mod, unit, vec=False):
    """`mod`: shift / scale are applied (else plain RMSNorm with weight); `unit`: their
    column stride is 1; `vec` (bm = 1 only): the row moves through shared memory with
    tile copies (vectorized global loads and stores)."""
    n = T.dynamic("n")
    lh = T.dynamic("lh")
    lc = T.dynamic("lc")

    @T.prim_func
    def main(
        x: T.Tensor((n, d), dtype),
        w: T.Tensor((d,), F32),
        shift: T.Tensor((lh,), F32),
        scale: T.Tensor((lc,), F32),
        seg: T.Tensor((n,), I32),
        y: T.Tensor((n, d), dtype),
        rstd: T.Tensor((n,), F32),
        eps: T.float32,
        hr: T.int32,
        hc: T.int32,
        cr: T.int32,
        cc: T.int32,
    ):
        with T.Kernel(T.ceildiv(n, bm), threads=threads) as bx:
            xf = T.alloc_fragment((bm, d), F32)
            sq = T.alloc_fragment((bm, d), F32)
            ss = T.alloc_fragment((bm,), F32)
            rs = T.alloc_fragment((bm,), F32)
            if vec:
                xs = T.alloc_shared((bm, d), dtype)
                ys = T.alloc_shared((bm, d), dtype)
                T.copy(x[bx * bm, 0], xs)
            for i, j in T.Parallel(bm, d):
                r = T.min(bx * bm + i, n - 1)
                if vec:
                    xf[i, j] = T.cast(xs[i, j], F32)
                else:
                    xf[i, j] = T.cast(x[r, j], F32)
                sq[i, j] = xf[i, j] * xf[i, j]
            T.reduce_sum(sq, ss, dim=1)
            for i in T.Parallel(bm):
                rs[i] = 1.0 / T.sqrt(ss[i] / d + eps)
            for i in T.Parallel(bm):
                if bx * bm + i < n:
                    rstd[bx * bm + i] = rs[i]
            for i, j in T.Parallel(bm, d):
                r = T.min(bx * bm + i, n - 1)
                v = xf[i, j] * rs[i] * w[j]
                if mod:
                    s = seg[r]
                    if unit:
                        v = v * (1.0 + scale[s * cr + j]) + shift[s * hr + j]
                    else:
                        v = v * (1.0 + scale[s * cr + j * cc]) + shift[s * hr + j * hc]
                if vec:
                    ys[i, j] = T.cast(v, dtype)
                elif bx * bm + i < n:
                    y[bx * bm + i, j] = T.cast(v, dtype)
            if vec:
                T.copy(ys, y[bx * bm, 0])

    return main


def _bwd_prim(d, bm, threads, dtype, unit):
    n = T.dynamic("n")
    lc = T.dynamic("lc")
    tt = T.dynamic("t")

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
        part: T.Tensor((tt, bm, 2, d), F32),
        cr: T.int32,
        cc: T.int32,
    ):
        with T.Kernel(tt, threads=threads) as t:
            acc_gx = T.alloc_fragment((bm, d), F32)
            acc_g = T.alloc_fragment((bm, d), F32)
            xh = T.alloc_fragment((bm, d), F32)
            gm = T.alloc_fragment((bm, d), F32)
            prod = T.alloc_fragment((bm, d), F32)
            c = T.alloc_fragment((bm,), F32)
            rs = T.alloc_fragment((bm,), F32)
            sb = tseg[t]
            start = tstart[t]
            stop = tstop[t]
            T.clear(acc_gx)
            T.clear(acc_g)
            for it in T.serial(T.ceildiv(stop - start, bm)):
                for i in T.Parallel(bm):
                    r = start + it * bm + i
                    rs[i] = T.if_then_else(r < stop, rstd[T.min(r, n - 1)], 0.0)
                for i, j in T.Parallel(bm, d):
                    r = start + it * bm + i
                    ok = r < stop
                    rr = T.min(r, n - 1)
                    gv = T.if_then_else(ok, T.cast(g[rr, j], F32), 0.0)
                    xv = T.if_then_else(ok, T.cast(x[rr, j], F32), 0.0) * rs[i]
                    if unit:
                        mj = w[j] * (1.0 + scale[sb * cr + j])
                    else:
                        mj = w[j] * (1.0 + scale[sb * cr + j * cc])
                    xh[i, j] = xv
                    gm[i, j] = gv * mj
                    prod[i, j] = gv * mj * xv
                    acc_gx[i, j] = acc_gx[i, j] + gv * xv
                    acc_g[i, j] = acc_g[i, j] + gv
                T.reduce_sum(prod, c, dim=1)
                for i, j in T.Parallel(bm, d):
                    r = start + it * bm + i
                    if r < stop:
                        dx[r, j] = T.cast(
                            rs[i] * (gm[i, j] - xh[i, j] * c[i] / d), dtype
                        )
            for i, j in T.Parallel(bm, d):
                part[t, i, 0, j] = acc_gx[i, j]
                part[t, i, 1, j] = acc_g[i, j]

    return main


def _bwd_dx_prim(d, bm, threads, dtype, unit, vec=False):
    """dx only (no parameter gradients): rows in parallel, modulation gathered per row
    (`unit`: unit column stride; `vec`, bm = 1 only: rows through shared memory)."""
    n = T.dynamic("n")
    lc = T.dynamic("lc")

    @T.prim_func
    def main(
        g: T.Tensor((n, d), dtype),
        x: T.Tensor((n, d), dtype),
        rstd: T.Tensor((n,), F32),
        w: T.Tensor((d,), F32),
        scale: T.Tensor((lc,), F32),
        seg: T.Tensor((n,), I32),
        dx: T.Tensor((n, d), dtype),
        cr: T.int32,
        cc: T.int32,
    ):
        with T.Kernel(T.ceildiv(n, bm), threads=threads) as bx:
            xh = T.alloc_fragment((bm, d), F32)
            gm = T.alloc_fragment((bm, d), F32)
            prod = T.alloc_fragment((bm, d), F32)
            c = T.alloc_fragment((bm,), F32)
            if vec:
                xs = T.alloc_shared((bm, d), dtype)
                gs = T.alloc_shared((bm, d), dtype)
                T.copy(x[bx * bm, 0], xs)
                T.copy(g[bx * bm, 0], gs)
            for i, j in T.Parallel(bm, d):
                rr = T.min(bx * bm + i, n - 1)
                if vec:
                    xv = T.cast(xs[i, j], F32)
                    gv = T.cast(gs[i, j], F32)
                else:
                    xv = T.cast(x[rr, j], F32)
                    gv = T.cast(g[rr, j], F32)
                xh[i, j] = xv * rstd[rr]
                if unit:
                    sj = scale[seg[rr] * cr + j]
                else:
                    sj = scale[seg[rr] * cr + j * cc]
                gm[i, j] = gv * w[j] * (1.0 + sj)
                prod[i, j] = gm[i, j] * xh[i, j]
            T.reduce_sum(prod, c, dim=1)
            for i, j in T.Parallel(bm, d):
                row = bx * bm + i
                rr = T.min(row, n - 1)
                v = T.cast(rstd[rr] * (gm[i, j] - xh[i, j] * c[i] / d), dtype)
                if vec:
                    xs[i, j] = v
                elif row < n:
                    dx[row, j] = v
            if vec:
                T.copy(xs, dx[bx * bm, 0])

    return main


def _grads_prim(d, bd, threads):
    """d_weight [d], d_shift [b, d], d_scale [b, d] (fp32) from the per-sample sums
    [2 (S_gx, S_g), b, d]; d_weight sums over samples."""
    b = T.dynamic("b")
    lc = T.dynamic("lc")

    @T.prim_func
    def main(
        sums: T.Tensor((2, b, d), F32),
        w: T.Tensor((d,), F32),
        scale: T.Tensor((lc,), F32),
        dw: T.Tensor((d,), F32),
        dshift: T.Tensor((b, d), F32),
        dscale: T.Tensor((b, d), F32),
        cr: T.int32,
        cc: T.int32,
    ):
        with T.Kernel(T.ceildiv(d, bd), threads=threads) as cb:
            acc = T.alloc_fragment((bd,), F32)
            T.clear(acc)
            for sb in T.serial(b):
                for j in T.Parallel(bd):
                    col = T.min(cb * bd + j, d - 1)
                    gx = sums[0, sb, col]
                    acc[j] = acc[j] + gx * (1.0 + scale[sb * cr + col * cc])
                    if cb * bd + j < d:
                        dshift[sb, col] = sums[1, sb, col]
                        dscale[sb, col] = w[col] * gx
            for j in T.Parallel(bd):
                if cb * bd + j < d:
                    dw[cb * bd + j] = acc[j]

    return main


FWD_CFGS = [
    {"bm": bm, "threads": th}
    for bm, th in ((1, 128), (2, 256), (1, 256), (4, 256), (2, 128))
]
VEC_CFGS = FWD_CFGS + [{"bm": 1, "threads": th, "vec": True} for th in (128, 256, 512)]
BWD_CFGS = [
    {"bm": bm, "threads": th, "per_sm": ps}
    for bm, th, ps in (
        (1, 128, 8),
        (1, 256, 4),
        (2, 128, 4),
        (1, 128, 4),
        (2, 256, 2),
        (1, 128, 16),
    )
]


def _batch(shift: Tensor | None, scale: Tensor | None) -> int:
    """Sample count of the per-sample modulation (1 without one)."""
    if shift is not None:
        return shift.shape[0]
    return scale.shape[0] if scale is not None else 1


def _weight(w: Tensor | None, d: int, device) -> Tensor:
    """The norm weight as contiguous fp32, ones when absent."""
    return f32(w, (d,), device) if w is not None else torch.ones(d, device=device)


def _launch_forward(x, w, shift, scale, seg, eps):
    n, d = x.shape
    b = _batch(shift, scale)
    dev = x.device
    wf = _weight(w, d, dev)
    sh = per_sample(shift, b, d, dev)
    sc = per_sample(scale, b, d, dev)
    sg = segment_ids(seg, n, dev)
    y = torch.empty_like(x)
    rstd = torch.empty(n, device=dev, dtype=torch.float32)
    mod = shift is not None or scale is not None
    unit = sh[2] == 1 and sc[2] == 1
    flags = (DTYPE[x.dtype], mod, unit)

    def run(c):
        vec = c.get("vec", False)
        kernel = compiled(_fwd_prim, dev, d, c["bm"], c["threads"], *flags, vec)
        kernel(x, wf, sh[0], sc[0], sg, y, rstd, float(eps), *sh[1:], *sc[1:])

    key = f"d{d}-{x.dtype}-{mod}{unit}"
    tuned_run("tilelang_rms_norm_fwd", dev, key, VEC_CFGS, run)
    return y, rstd


def _launch_backward(g, x, rstd, w, scale, seg, b, need_w, need_shift, need_scale):
    n, d = x.shape
    dev = x.device
    wf = _weight(w, d, dev)
    dx = torch.empty_like(x)
    sc = per_sample(scale, b, d, dev)
    flags = (DTYPE[x.dtype], sc[2] == 1)
    if not (need_w or need_shift or need_scale):
        sg = segment_ids(seg, n, dev)

        def plain(c):
            vec = c.get("vec", False)
            kernel = compiled(_bwd_dx_prim, dev, d, c["bm"], c["threads"], *flags, vec)
            kernel(g, x, rstd, wf, sc[0], sg, dx, *sc[1:])

        key = f"d{d}-{x.dtype}-{flags[1]}"
        tuned_run("tilelang_rms_norm_bwd_dx", dev, key, VEC_CFGS, plain)
        return dx, x.new_empty(0), x.new_empty(0), x.new_empty(0)

    def run(c):
        plan = tile_plan(seg, n, b, c["bm"], dev, c["per_sm"])
        part = torch.empty(plan.tiles, c["bm"], 2, d, device=dev, dtype=torch.float32)
        kernel = compiled(_bwd_prim, dev, d, c["bm"], c["threads"], *flags)
        kernel(
            g, x, rstd, wf, sc[0], plan.tseg, plan.tstart, plan.tstop, dx, part,
            *sc[1:],
        )  # fmt: skip
        return plan, part

    key = f"d{d}-{x.dtype}-{seg is not None}-{flags[1]}"
    plan, part = tuned_run("tilelang_rms_norm_bwd", dev, key, BWD_CFGS, run)
    sums = segment_sums(part, plan)
    fp32 = dict(device=dev, dtype=torch.float32)
    dw = torch.empty(d, **fp32)
    dshift = torch.empty(plan.b, d, **fp32)
    dscale = torch.empty(plan.b, d, **fp32)
    compiled(_grads_prim, dev, d, 128, 128)(
        sums, wf, sc[0], dw, dshift, dscale, *sc[1:]
    )
    return (
        dx,
        dw.to(w.dtype) if need_w else x.new_empty(0),
        dshift if need_shift else x.new_empty(0),
        dscale if need_scale else x.new_empty(0),
    )


@custom_op("kohakumoss::tilelang_rms_norm_fwd", mutates_args=())
def _fwd_op(
    x: Tensor,
    weight: Tensor | None,
    shift: Tensor | None,
    scale: Tensor | None,
    seg: Tensor | None,
    eps: float,
) -> tuple[Tensor, Tensor]:
    return _launch_forward(x, weight, shift, scale, seg, eps)


@_fwd_op.register_fake
def _(x, weight, shift, scale, seg, eps):
    return torch.empty_like(x), x.new_empty(x.shape[0], dtype=torch.float32)


@custom_op("kohakumoss::tilelang_rms_norm_bwd", mutates_args=())
def _bwd_op(
    g: Tensor,
    x: Tensor,
    rstd: Tensor,
    weight: Tensor | None,
    scale: Tensor | None,
    seg: Tensor | None,
    batch: int,
    need_w: bool,
    need_shift: bool,
    need_scale: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _launch_backward(
        g, x, rstd, weight, scale, seg, batch, need_w, need_shift, need_scale
    )


@_bwd_op.register_fake
def _(g, x, rstd, weight, scale, seg, batch, need_w, need_shift, need_scale):
    d = x.shape[1]
    dw = x.new_empty(d, dtype=weight.dtype) if need_w else x.new_empty(0)
    ds = x.new_empty(batch, d, dtype=torch.float32) if need_shift else x.new_empty(0)
    dc = x.new_empty(batch, d, dtype=torch.float32) if need_scale else x.new_empty(0)
    return torch.empty_like(x), dw, ds, dc


def _setup(ctx, inputs, output):
    x, weight, shift, scale, seg, _ = inputs
    ctx.save_for_backward(x, output[1], weight, scale, seg)
    ctx.shift_dtype = None if shift is None else shift.dtype
    ctx.batch = _batch(shift, scale)


def _backward(ctx, gy, _grstd):
    x, rstd, weight, scale, seg = ctx.saved_tensors
    need = ctx.needs_input_grad
    dx, dw, dshift, dscale = _bwd_op(
        gy.contiguous(), x, rstd, weight, scale, seg, ctx.batch, *need[1:4]
    )
    return (
        dx,
        dw if need[1] else None,
        dshift.to(ctx.shift_dtype) if need[2] else None,
        dscale.to(scale.dtype) if need[3] else None,
        None,
        None,
    )


_fwd_op.register_autograd(_backward, setup_context=_setup)


def rms_norm_mod(
    x: Tensor,
    weight: Tensor | None,
    shift: Tensor | None,
    scale: Tensor | None,
    seg: Tensor | None,
    eps: float,
) -> Tensor:
    """`rms_norm(x) * (1 + scale[seg]) + shift[seg]` over the last dim, as the eager
    reference computes it."""
    shape = x.shape
    try:
        y, _ = _fwd_op(
            x.reshape(-1, shape[-1]).contiguous(), weight, shift, scale, seg, eps
        )
    except CompileError:
        return eager_norm.rms_norm_mod(x, weight, shift, scale, seg, eps)
    return y.view(shape)


def rms_norm(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """`x / rms(x) * weight` over the last dim, as the eager reference computes it."""
    return rms_norm_mod(x, weight, None, None, None, eps)


register(
    "rms_norm",
    "tilelang",
    kernel_or_compile("tilelang", "rms_norm", rms_norm, shape_key),
    FP16_SUPPORTS,
)
register(
    "rms_norm_mod",
    "tilelang",
    kernel_or_compile("tilelang", "rms_norm_mod", rms_norm_mod, shape_key),
    FP16_SUPPORTS,
)
