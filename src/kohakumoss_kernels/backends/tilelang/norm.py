"""TileLang RMSNorm and RMSNorm + per-sample modulation, forward and backward.

Same algorithm as the Triton twin: forward holds `[BM, D]` rows in a fragment and
keeps rstd; backward runs one program per segment tile, writes dx and per-tile
fp32-pair column partials of sum(g * xhat) and sum(g); pair segment sums and an
epilogue give the parameter gradients. Small parameters (weight, shift, scale) are
passed as fp32. See docs/internals/fp16.md.
"""

import tilelang.language as T
import torch
from torch import Tensor
from torch.library import custom_op

from kohakumoss_kernels.backends.compile.race import raced, shape_key
from kohakumoss_kernels.backends.eager import norm as eager_norm
from kohakumoss_kernels.backends.tilelang._blockops import (
    DTYPE,
    CompileError,
    compiled,
    f32,
    per_sample,
    segment_sums,
    tile_plan,
    tuned_run,
)
from kohakumoss_kernels.backends.tilelang._norm_df import bwd_df_prim, grads_prim
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.reduce import DF, MODE
from kohakumoss_kernels.registry import register

__all__ = ["rms_norm", "rms_norm_mod"]

F32 = "float32"
I32 = "int32"


def _fwd_prim(d, bm, threads, dtype):
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
            for i, j in T.Parallel(bm, d):
                r = T.min(bx * bm + i, n - 1)
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
                s = seg[r]
                v = xf[i, j] * rs[i] * w[j] * (1.0 + scale[s * cr + j * cc])
                v = v + shift[s * hr + j * hc]
                if bx * bm + i < n:
                    y[bx * bm + i, j] = T.cast(v, dtype)

    return main


def _bwd_prim(d, bm, threads, dtype):
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
        part: T.Tensor((tt, bm, 2, 2, d), F32),
        cr: T.int32,
        cc: T.int32,
    ):
        with T.Kernel(tt, threads=threads) as t:
            acc_gx = T.alloc_fragment((bm, d), F32)
            acc_g = T.alloc_fragment((bm, d), F32)
            c_gx = T.alloc_fragment((bm, d), F32)
            c_g = T.alloc_fragment((bm, d), F32)
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
            T.clear(c_gx)
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
                    xv = T.if_then_else(ok, T.cast(x[rr, j], F32), 0.0) * rs[i]
                    mj = w[j] * (1.0 + scale[sb * cr + j * cc])
                    xh[i, j] = xv
                    gm[i, j] = gv * mj
                    prod[i, j] = gv * mj * xv
                    y1 = gv * xv - c_gx[i, j]
                    t1 = acc_gx[i, j] + y1
                    c_gx[i, j] = (t1 - acc_gx[i, j]) - y1
                    acc_gx[i, j] = t1
                    y2 = gv - c_g[i, j]
                    t2 = acc_g[i, j] + y2
                    c_g[i, j] = (t2 - acc_g[i, j]) - y2
                    acc_g[i, j] = t2
                T.reduce_sum(prod, c, dim=1)
                for i, j in T.Parallel(bm, d):
                    r = start + it * bm + i
                    if r < stop:
                        dx[r, j] = T.cast(
                            rs[i] * (gm[i, j] - xh[i, j] * c[i] / d), dtype
                        )
            for i, j in T.Parallel(bm, d):
                part[t, i, 0, 0, j] = acc_gx[i, j]
                part[t, i, 0, 1, j] = -c_gx[i, j]
                part[t, i, 1, 0, j] = acc_g[i, j]
                part[t, i, 1, 1, j] = -c_g[i, j]

    return main


def _bwd_dx_prim(d, bm, threads, dtype):
    """dx only (no parameter gradients): rows in parallel, modulation gathered per row."""
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
            for i, j in T.Parallel(bm, d):
                rr = T.min(bx * bm + i, n - 1)
                xh[i, j] = T.cast(x[rr, j], F32) * rstd[rr]
                sj = scale[seg[rr] * cr + j * cc]
                gm[i, j] = T.cast(g[rr, j], F32) * w[j] * (1.0 + sj)
                prod[i, j] = gm[i, j] * xh[i, j]
            T.reduce_sum(prod, c, dim=1)
            for i, j in T.Parallel(bm, d):
                row = bx * bm + i
                rr = T.min(row, n - 1)
                if row < n:
                    dx[row, j] = T.cast(
                        rstd[rr] * (gm[i, j] - xh[i, j] * c[i] / d), dtype
                    )

    return main


FWD_CFGS = [
    {"bm": bm, "threads": th}
    for bm, th in ((1, 128), (2, 256), (1, 256), (4, 256), (2, 128))
]
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


def _forward(x, w, shift, scale, seg, eps):
    n, d = x.shape
    b = (
        shift.shape[0]
        if shift is not None
        else scale.shape[0] if scale is not None else 1
    )
    dev = x.device
    wf = f32(w, (d,), dev) if w is not None else torch.ones(d, device=dev)
    sh = per_sample(shift, b, d, dev)
    sc = per_sample(scale, b, d, dev)
    sg = (
        seg.to(torch.int32).contiguous()
        if seg is not None
        else torch.zeros(n, device=dev, dtype=torch.int32)
    )
    y = torch.empty_like(x)
    rstd = torch.empty(n, device=dev, dtype=torch.float32)

    def run(c):
        kernel = compiled(_fwd_prim, dev, d, c["bm"], c["threads"], DTYPE[x.dtype])
        kernel(x, wf, sh[0], sc[0], sg, y, rstd, float(eps), *sh[1:], *sc[1:])

    tuned_run("tilelang_rms_norm_fwd", dev, f"d{d}-{x.dtype}", FWD_CFGS, run)
    return y, rstd


def _backward(g, x, rstd, w, scale, seg, b, need_w, need_shift, need_scale, eps):
    n, d = x.shape
    dev = x.device
    wf = f32(w, (d,), dev) if w is not None else torch.ones(d, device=dev)
    dx = torch.empty_like(x)
    none = x.new_empty(0)
    sc = per_sample(scale, b, d, dev)
    if not (need_w or need_shift or need_scale):
        sg = (
            seg.to(torch.int32).contiguous()
            if seg is not None
            else torch.zeros(n, device=dev, dtype=torch.int32)
        )

        def plain(c):
            kernel = compiled(
                _bwd_dx_prim, dev, d, c["bm"], c["threads"], DTYPE[x.dtype]
            )
            kernel(g, x, rstd, wf, sc[0], sg, dx, *sc[1:])

        tuned_run("tilelang_rms_norm_bwd_dx", dev, f"d{d}-{x.dtype}", FWD_CFGS, plain)
        return dx, none, x.new_empty(0), x.new_empty(0)

    prim = bwd_df_prim if DF else _bwd_prim
    extra = (float(eps),) if DF else ()

    def run(c):
        plan = tile_plan(seg, n, b, c["bm"], dev, c["per_sm"])
        part = torch.empty(
            plan.tiles, c["bm"], 2, 2, d, device=dev, dtype=torch.float32
        )
        kernel = compiled(prim, dev, d, c["bm"], c["threads"], DTYPE[x.dtype])
        kernel(
            g, x, rstd, wf, sc[0], plan.tseg, plan.tstart, plan.tstop, dx, part,
            *sc[1:], *extra,
        )  # fmt: skip
        return plan, part

    key = f"d{d}-{x.dtype}-{seg is not None}-{MODE}"
    plan, part = tuned_run("tilelang_rms_norm_bwd", dev, key, BWD_CFGS, run)
    sums = segment_sums(part, plan, rounded=False)
    fp32 = dict(device=dev, dtype=torch.float32)
    dw = torch.empty(d, **fp32)
    dshift = torch.empty(plan.b, d, **fp32)
    dscale = torch.empty(plan.b, d, **fp32)
    compiled(grads_prim, dev, d, 128, 128)(
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
    return _forward(x, weight, shift, scale, seg, eps)


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
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _backward(
        g, x, rstd, weight, scale, seg, batch, need_w, need_shift, need_scale, eps
    )


@_bwd_op.register_fake
def _(g, x, rstd, weight, scale, seg, batch, need_w, need_shift, need_scale, eps):
    d = x.shape[1]
    dw = x.new_empty(d, dtype=weight.dtype) if need_w else x.new_empty(0)
    ds = x.new_empty(batch, d, dtype=torch.float32) if need_shift else x.new_empty(0)
    dc = x.new_empty(batch, d, dtype=torch.float32) if need_scale else x.new_empty(0)
    return torch.empty_like(x), dw, ds, dc


def _setup(ctx, inputs, output):
    x, weight, shift, scale, seg, eps = inputs
    ctx.save_for_backward(x, output[1], weight, scale, seg)
    ctx.eps = eps
    ctx.shift_dtype = None if shift is None else shift.dtype
    ctx.batch = (
        shift.shape[0]
        if shift is not None
        else scale.shape[0] if scale is not None else 1
    )


def _backward_rule(ctx, gy, _grstd):
    x, rstd, weight, scale, seg = ctx.saved_tensors
    need = ctx.needs_input_grad
    dx, dw, dshift, dscale = _bwd_op(
        gy.contiguous(),
        x,
        rstd,
        weight,
        scale,
        seg,
        ctx.batch,
        need[1],
        need[2],
        need[3],
        ctx.eps,
    )
    return (
        dx,
        dw if need[1] else None,
        dshift.to(ctx.shift_dtype) if need[2] else None,
        dscale.to(scale.dtype) if need[3] else None,
        None,
        None,
    )


_fwd_op.register_autograd(_backward_rule, setup_context=_setup)


def rms_norm_mod(
    x: Tensor,
    weight: Tensor | None,
    shift: Tensor | None,
    scale: Tensor | None,
    seg: Tensor | None,
    eps: float,
) -> Tensor:
    """`rms_norm(x) * (1 + scale[seg]) + shift[seg]` over the last dim (eager spec)."""
    shape = x.shape
    try:
        y, _ = _fwd_op(
            x.reshape(-1, shape[-1]).contiguous(), weight, shift, scale, seg, eps
        )
    except CompileError:
        return eager_norm.rms_norm_mod(x, weight, shift, scale, seg, eps)
    return y.view(shape)


def rms_norm(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """`x / rms(x) * weight` over the last dim (eager spec)."""
    return rms_norm_mod(x, weight, None, None, None, eps)


_SUPPORTS = requires(kinds=("cuda",), min_sm=70, dtypes=(torch.float16,))
register(
    "rms_norm",
    "tilelang",
    raced("tilelang", "rms_norm", rms_norm, shape_key),
    _SUPPORTS,
)
register(
    "rms_norm_mod",
    "tilelang",
    raced("tilelang", "rms_norm_mod", rms_norm_mod, shape_key),
    _SUPPORTS,
)
