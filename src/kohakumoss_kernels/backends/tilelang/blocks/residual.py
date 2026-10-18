"""TileLang gated residual `x * r + o * gate[seg]`, forward and backward.

The backward runs one program per segment tile, writes d_o (and d_x when r != 1) and
per-tile fp32 partials of sum(g * o) that are summed per sample into d_gate.
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
    per_sample,
    segment_ids,
    segment_sums,
    tile_plan,
    tuned_run,
)
from kohakumoss_kernels.registry import register

__all__ = ["gated_residual"]

FWD_BM, FWD_THREADS = 4, 128


def _fwd_prim(d, bm, threads, gated, dtype, xdt, unit=True):
    """y = x * r + o * gate[seg]: o in `dtype`, x and y in `xdt` (the residual
    stream, fp16 or fp32); `unit`: the gate's column stride is 1."""
    n = T.dynamic("n")
    lg = T.dynamic("lg")

    @T.prim_func
    def with_gate(
        x: T.Tensor((n, d), xdt),
        o: T.Tensor((n, d), dtype),
        gate: T.Tensor((lg,), F32),
        seg: T.Tensor((n,), I32),
        y: T.Tensor((n, d), xdt),
        r: T.float32,
        gr: T.int32,
        gc: T.int32,
    ):
        with T.Kernel(T.ceildiv(n, bm), threads=threads) as bx:
            for i, j in T.Parallel(bm, d):
                row = bx * bm + i
                rr = T.min(row, n - 1)
                if unit:
                    gv = gate[seg[rr] * gr + j]
                else:
                    gv = gate[seg[rr] * gr + j * gc]
                v = T.cast(x[rr, j], F32) * r + T.cast(o[rr, j], F32) * gv
                if row < n:
                    y[row, j] = T.cast(v, xdt)

    @T.prim_func
    def plain(
        x: T.Tensor((n, d), xdt),
        o: T.Tensor((n, d), dtype),
        y: T.Tensor((n, d), xdt),
        r: T.float32,
    ):
        with T.Kernel(T.ceildiv(n, bm), threads=threads) as bx:
            for i, j in T.Parallel(bm, d):
                row = bx * bm + i
                rr = T.min(row, n - 1)
                v = T.cast(x[rr, j], F32) * r + T.cast(o[rr, j], F32)
                if row < n:
                    y[row, j] = T.cast(v, xdt)

    return with_gate if gated else plain


def _bwd_prim(d, bm, bd, threads, need_dx, dtype, xdt, unit):
    """Backward with per-tile fp32 partials of sum(g * o); g and dx in `xdt`, o and
    do in `dtype`; `unit`: the gate's column stride is 1."""
    n = T.dynamic("n")
    lg = T.dynamic("lg")
    tt = T.dynamic("t")

    @T.prim_func
    def main(
        g: T.Tensor((n, d), xdt),
        o: T.Tensor((n, d), dtype),
        gate: T.Tensor((lg,), F32),
        tseg: T.Tensor((tt,), I32),
        tstart: T.Tensor((tt,), I32),
        tstop: T.Tensor((tt,), I32),
        dx: T.Tensor((n, d), xdt),
        do: T.Tensor((n, d), dtype),
        part: T.Tensor((tt, bm, 1, d), F32),
        r: T.float32,
        gr: T.int32,
        gc: T.int32,
    ):
        with T.Kernel(tt, T.ceildiv(d, bd), threads=threads) as (t, cb):
            acc = T.alloc_fragment((bm, bd), F32)
            gf = T.alloc_fragment((bm, bd), F32)
            sb = tseg[t]
            start = tstart[t]
            stop = tstop[t]
            T.clear(acc)
            for it in T.serial(T.ceildiv(stop - start, bm)):
                for i, j in T.Parallel(bm, bd):
                    row = start + it * bm + i
                    col = cb * bd + j
                    ok = row < stop and col < d
                    rr = T.min(row, n - 1)
                    cc = T.min(col, d - 1)
                    gf[i, j] = T.if_then_else(ok, T.cast(g[rr, cc], F32), 0.0)
                    ov = T.if_then_else(ok, T.cast(o[rr, cc], F32), 0.0)
                    acc[i, j] = acc[i, j] + gf[i, j] * ov
                for i, j in T.Parallel(bm, bd):
                    row = start + it * bm + i
                    col = cb * bd + j
                    if row < stop and col < d:
                        if unit:
                            gv = gate[sb * gr + col]
                        else:
                            gv = gate[sb * gr + col * gc]
                        do[row, col] = T.cast(gf[i, j] * gv, dtype)
                        if need_dx:
                            dx[row, col] = T.cast(gf[i, j] * r, xdt)
            for i, j in T.Parallel(bm, bd):
                if cb * bd + j < d:
                    part[t, i, 0, cb * bd + j] = acc[i, j]

    return main


def _bwd_plain_prim(d, bm, threads, need_dx, dtype, xdt, unit):
    """d_o = g * gate[seg] (and d_x = g * r) when no gate gradient is needed."""
    n = T.dynamic("n")
    lg = T.dynamic("lg")

    @T.prim_func
    def main(
        g: T.Tensor((n, d), xdt),
        gate: T.Tensor((lg,), F32),
        seg: T.Tensor((n,), I32),
        dx: T.Tensor((n, d), xdt),
        do: T.Tensor((n, d), dtype),
        r: T.float32,
        gr: T.int32,
        gc: T.int32,
    ):
        with T.Kernel(T.ceildiv(n, bm), threads=threads) as bx:
            for i, j in T.Parallel(bm, d):
                row = bx * bm + i
                rr = T.min(row, n - 1)
                gv = T.cast(g[rr, j], F32)
                if unit:
                    gt = gate[seg[rr] * gr + j]
                else:
                    gt = gate[seg[rr] * gr + j * gc]
                if row < n:
                    do[row, j] = T.cast(gv * gt, dtype)
                    if need_dx:
                        dx[row, j] = T.cast(gv * r, xdt)

    return main


def residual_forward(x, o, gate, seg, r):
    """x * r + o * gate[seg] for x, o [N, D] (no autograd); y in x's dtype."""
    n, d = x.shape
    y = torch.empty_like(x)
    tile = (d, FWD_BM, FWD_THREADS)
    dts = (DTYPE[o.dtype], DTYPE[x.dtype])
    if gate is None:
        kernel = compiled(_fwd_prim, x.device, *tile, False, *dts)
        kernel(x, o, y, float(r))
    else:
        flat, gr, gc = per_sample(gate, *gate.shape, x.device)
        kernel = compiled(_fwd_prim, x.device, *tile, True, *dts, gc == 1)
        kernel(x, o, flat, segment_ids(seg, n, x.device), y, float(r), gr, gc)
    return y


PLAIN_CFGS = [
    {"bm": bm, "threads": th} for bm, th in ((1, 128), (2, 256), (1, 256), (4, 256))
]
BWD_CFGS = [
    {"bm": bm, "bd": bd, "threads": th, "per_sm": ps}
    for bm, bd, th, ps in (
        (4, 256, 128, 2), (2, 512, 128, 2), (4, 512, 256, 1), (8, 256, 256, 1), (2, 256, 128, 4)
    )
]  # fmt: skip


def residual_backward(g, o, gate, seg, r, need_gate=True):
    """(d_x or empty when r == 1, d_o in o's dtype, d_gate fp32 [B, D] or empty) (no
    autograd); g is in x's dtype, o may be empty (of its dtype) without `need_gate`."""
    n, d = g.shape
    dev = g.device
    need_dx = r != 1.0
    do = torch.empty_like(g, dtype=o.dtype)
    dx = torch.empty_like(g) if need_dx else g
    gf, gr, gc = per_sample(gate, *gate.shape, dev)
    dts = (DTYPE[o.dtype], DTYPE[g.dtype], gc == 1)
    if not need_gate:
        sg = segment_ids(seg, n, dev)

        def plain(c):
            kernel = compiled(
                _bwd_plain_prim, dev, d, c["bm"], c["threads"], need_dx, *dts
            )
            kernel(g, gf, sg, dx, do, float(r), gr, gc)

        key = f"d{d}-{g.dtype}-{o.dtype}-{need_dx}-{dts[2]}"
        tuned_run("tilelang_gated_residual_bwd_plain", dev, key, PLAIN_CFGS, plain)
        return (dx if need_dx else g.new_empty(0)), do, g.new_empty(0)

    def run(c):
        plan = tile_plan(seg, n, gate.shape[0], c["bm"], dev, c["per_sm"])
        part = torch.empty(plan.tiles, c["bm"], 1, d, device=dev, dtype=torch.float32)
        kernel = compiled(
            _bwd_prim, dev, d, c["bm"], c["bd"], c["threads"], need_dx, *dts
        )
        kernel(
            g, o, gf, plan.tseg, plan.tstart, plan.tstop, dx, do, part, float(r), gr, gc
        )
        return plan, part

    key = f"d{d}-{g.dtype}-{o.dtype}-{need_dx}-{dts[2]}"
    plan, part = tuned_run("tilelang_gated_residual_bwd", dev, key, BWD_CFGS, run)
    dgate = segment_sums(part, plan)[0]
    return (dx if need_dx else g.new_empty(0)), do, dgate


@custom_op("kohakumoss::tilelang_gated_residual_fwd", mutates_args=())
def _fwd_op(
    x: Tensor, o: Tensor, gate: Tensor | None, seg: Tensor | None, r: float
) -> Tensor:
    return residual_forward(x, o, gate, seg, r)


@_fwd_op.register_fake
def _(x, o, gate, seg, r):
    return torch.empty_like(x)


@custom_op("kohakumoss::tilelang_gated_residual_bwd", mutates_args=())
def _bwd_op(
    g: Tensor, o: Tensor, gate: Tensor, seg: Tensor, r: float, need_gate: bool
) -> tuple[Tensor, Tensor, Tensor]:
    return residual_backward(g, o, gate, seg, r, need_gate)


@_bwd_op.register_fake
def _(g, o, gate, seg, r, need_gate):
    dx = torch.empty_like(g) if r != 1.0 else g.new_empty(0)
    dgate = (
        gate.new_empty(gate.shape, dtype=torch.float32) if need_gate else g.new_empty(0)
    )
    return dx, torch.empty_like(g, dtype=o.dtype), dgate


def _setup(ctx, inputs, output):
    x, o, gate, seg, r = inputs
    gate_grad = gate is not None and ctx.needs_input_grad[2]
    ctx.save_for_backward(o if gate_grad else o.new_empty(0), gate, seg)
    ctx.r = r
    ctx.x_dtype = x.dtype
    ctx.o_dtype = o.dtype


def _backward(ctx, gy):
    o, gate, seg = ctx.saved_tensors
    r = ctx.r
    need = ctx.needs_input_grad
    if gate is None:
        dx = gy if r == 1.0 else (gy.float() * r).to(ctx.x_dtype)
        return dx, gy.to(ctx.o_dtype), None, None, None
    dx, do, dgate = _bwd_op(gy.contiguous(), o, gate, seg, r, need[2])
    dx = gy if r == 1.0 else dx
    return dx, do, dgate.to(gate.dtype) if need[2] else None, None, None


_fwd_op.register_autograd(_backward, setup_context=_setup)


def gated_residual(
    x: Tensor, o: Tensor, gate: Tensor | None, seg: Tensor | None, r: float
) -> Tensor:
    """`x * r + o * gate[seg]`, as the eager reference computes it; x, o [N, D], gate
    [B, D] per sample."""
    shape = x.shape
    d = shape[-1]
    try:
        y = _fwd_op(
            x.reshape(-1, d).contiguous(),
            o.reshape(-1, d).contiguous(),
            gate,
            seg,
            float(r),
        )
    except CompileError:
        return eager_norm.gated_residual(x, o, gate, seg, r)
    return y.view(shape)


register(
    "gated_residual",
    "tilelang",
    kernel_or_compile("tilelang", "gated_residual", gated_residual, shape_key),
    FP16_SUPPORTS,
)
