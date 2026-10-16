"""TileLang gated residual `x * r + o * gate[seg]`, forward and backward.

The backward runs one program per segment tile, writes d_o (and d_x when r != 1) and
per-tile partials of sum(g * o) that are summed per sample into d_gate.
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
    segment_sums,
    tile_plan,
    tuned_run,
)
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.registry import register

__all__ = ["gated_residual"]

F32 = "float32"
I32 = "int32"
BD = 256


def _bm(d: int) -> int:
    return 4


def _threads(d: int) -> int:
    return 128


def _fwd_prim(d, bm, threads, gated, dtype):
    n = T.dynamic("n")
    b = T.dynamic("b")

    @T.prim_func
    def with_gate(
        x: T.Tensor((n, d), dtype),
        o: T.Tensor((n, d), dtype),
        gate: T.Tensor((b, d), F32),
        seg: T.Tensor((n,), I32),
        y: T.Tensor((n, d), dtype),
        r: T.float32,
    ):
        with T.Kernel(T.ceildiv(n, bm), threads=threads) as bx:
            for i, j in T.Parallel(bm, d):
                row = bx * bm + i
                rr = T.min(row, n - 1)
                v = T.cast(x[rr, j], F32) * r + T.cast(o[rr, j], F32) * gate[seg[rr], j]
                if row < n:
                    y[row, j] = T.cast(v, dtype)

    @T.prim_func
    def plain(
        x: T.Tensor((n, d), dtype),
        o: T.Tensor((n, d), dtype),
        y: T.Tensor((n, d), dtype),
        r: T.float32,
    ):
        with T.Kernel(T.ceildiv(n, bm), threads=threads) as bx:
            for i, j in T.Parallel(bm, d):
                row = bx * bm + i
                rr = T.min(row, n - 1)
                v = T.cast(x[rr, j], F32) * r + T.cast(o[rr, j], F32)
                if row < n:
                    y[row, j] = T.cast(v, dtype)

    return with_gate if gated else plain


def _bwd_prim(d, bm, bd, threads, need_dx, dtype):
    n = T.dynamic("n")
    b = T.dynamic("b")
    tt = T.dynamic("t")

    @T.prim_func
    def main(
        g: T.Tensor((n, d), dtype),
        o: T.Tensor((n, d), dtype),
        gate: T.Tensor((b, d), F32),
        tseg: T.Tensor((tt,), I32),
        tstart: T.Tensor((tt,), I32),
        tstop: T.Tensor((tt,), I32),
        dx: T.Tensor((n, d), dtype),
        do: T.Tensor((n, d), dtype),
        part: T.Tensor((tt, bm, 1, d), F32),
        r: T.float32,
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
                    acc[i, j] += gf[i, j] * T.if_then_else(
                        ok, T.cast(o[rr, cc], F32), 0.0
                    )
                for i, j in T.Parallel(bm, bd):
                    row = start + it * bm + i
                    col = cb * bd + j
                    if row < stop and col < d:
                        do[row, col] = T.cast(gf[i, j] * gate[sb, col], dtype)
                        if need_dx:
                            dx[row, col] = T.cast(gf[i, j] * r, dtype)
            for i, j in T.Parallel(bm, bd):
                if cb * bd + j < d:
                    part[t, i, 0, cb * bd + j] = acc[i, j]

    return main


def _bwd_plain_prim(d, bm, threads, need_dx, dtype):
    """d_o = g * gate[seg] (and d_x = g * r) when no gate gradient is needed."""
    n = T.dynamic("n")
    b = T.dynamic("b")

    @T.prim_func
    def main(
        g: T.Tensor((n, d), dtype),
        gate: T.Tensor((b, d), F32),
        seg: T.Tensor((n,), I32),
        dx: T.Tensor((n, d), dtype),
        do: T.Tensor((n, d), dtype),
        r: T.float32,
    ):
        with T.Kernel(T.ceildiv(n, bm), threads=threads) as bx:
            for i, j in T.Parallel(bm, d):
                row = bx * bm + i
                rr = T.min(row, n - 1)
                gv = T.cast(g[rr, j], F32)
                if row < n:
                    do[row, j] = T.cast(gv * gate[seg[rr], j], dtype)
                    if need_dx:
                        dx[row, j] = T.cast(gv * r, dtype)

    return main


def residual_forward(x, o, gate, seg, r):
    """x * r + o * gate[seg] for x, o [N, D] (no autograd)."""
    n, d = x.shape
    y = torch.empty_like(x)
    bm = _bm(d)
    kernel = compiled(
        _fwd_prim, x.device, d, bm, _threads(d), gate is not None, DTYPE[x.dtype]
    )
    if gate is None:
        kernel(x, o, y, float(r))
    else:
        kernel(
            x,
            o,
            f32(gate, gate.shape, x.device),
            seg.to(torch.int32).contiguous(),
            y,
            float(r),
        )
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
    """(d_x or empty when r == 1, d_o, d_gate fp32 [B, D] or empty) (no autograd)."""
    n, d = g.shape
    dev = g.device
    need_dx = r != 1.0
    do = torch.empty_like(g)
    dx = torch.empty_like(g) if need_dx else g
    gf = f32(gate, gate.shape, dev)
    if not need_gate:
        sg = seg.to(torch.int32).contiguous()

        def plain(c):
            kernel = compiled(
                _bwd_plain_prim, dev, d, c["bm"], c["threads"], need_dx, DTYPE[g.dtype]
            )
            kernel(g, gf, sg, dx, do, float(r))

        key = f"d{d}-{g.dtype}-{need_dx}"
        tuned_run("f6tl_gated_residual_bwd_plain", dev, key, PLAIN_CFGS, plain)
        return (dx if need_dx else g.new_empty(0)), do, g.new_empty(0)

    def run(c):
        plan = tile_plan(seg, n, gate.shape[0], c["bm"], dev, c["per_sm"])
        part = torch.empty(plan.tiles, c["bm"], 1, d, device=dev, dtype=torch.float32)
        kernel = compiled(
            _bwd_prim, dev, d, c["bm"], c["bd"], c["threads"], need_dx, DTYPE[g.dtype]
        )
        kernel(g, o, gf, plan.tseg, plan.tstart, plan.tstop, dx, do, part, float(r))
        return plan, part

    key = f"d{d}-{g.dtype}-{need_dx}"
    plan, part = tuned_run("f6tl_gated_residual_bwd", dev, key, BWD_CFGS, run)
    return (dx if need_dx else g.new_empty(0)), do, segment_sums(part, plan, 1, d)[0]


@custom_op("kohakumoss::f6tl_gated_residual_fwd", mutates_args=())
def _fwd_op(
    x: Tensor, o: Tensor, gate: Tensor | None, seg: Tensor | None, r: float
) -> Tensor:
    return residual_forward(x, o, gate, seg, r)


@_fwd_op.register_fake
def _(x, o, gate, seg, r):
    return torch.empty_like(x)


@custom_op("kohakumoss::f6tl_gated_residual_bwd", mutates_args=())
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
    return dx, torch.empty_like(g), dgate


def _setup(ctx, inputs, output):
    x, o, gate, seg, r = inputs
    ctx.save_for_backward(o, gate, seg)
    ctx.r = r
    ctx.x_dtype = x.dtype


def _backward(ctx, gy):
    o, gate, seg = ctx.saved_tensors
    r = ctx.r
    need = ctx.needs_input_grad
    if gate is None:
        dx = gy if r == 1.0 else (gy.float() * r).to(ctx.x_dtype)
        return dx, gy.to(o.dtype), None, None, None
    dx, do, dgate = _bwd_op(gy.contiguous(), o, gate, seg, r, need[2])
    dx = gy if r == 1.0 else dx
    return dx, do, dgate.to(gate.dtype) if need[2] else None, None, None


_fwd_op.register_autograd(_backward, setup_context=_setup)


def gated_residual(
    x: Tensor, o: Tensor, gate: Tensor | None, seg: Tensor | None, r: float
) -> Tensor:
    """`x * r + o * gate[seg]` (eager spec); x, o [N, D], gate [B, D] per sample."""
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
    raced("tilelang", "gated_residual", gated_residual, shape_key),
    requires(kinds=("cuda",), min_sm=70, dtypes=(torch.float16,)),
)
