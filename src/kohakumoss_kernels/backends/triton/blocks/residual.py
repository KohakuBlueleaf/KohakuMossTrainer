"""Triton gated residual `x * r + o * gate[seg]`, forward and backward.

The backward writes `d_o = g * gate[seg]` (and `d_x = g * r` when r != 1) per segment
tile and per-tile fp32 partials of `sum(g * o)`, summed per sample into `d_gate`.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op

from kohakumoss_kernels.backends.compile.choice import kernel_or_compile, shape_key
from kohakumoss_kernels.backends.triton._common import Launcher, eager_function
from kohakumoss_kernels.backends.triton.blocks._blockops import (
    SegPlan,
    reduce_candidates,
    row_candidates,
    strides2,
    tile_range,
)
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.buckets import cdiv, pow2
from kohakumoss_kernels.tuning.tuner import select

__all__ = ["gated_residual"]

DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _fwd_kernel(
    x_ptr,
    o_ptr,
    gate_ptr,
    seg_ptr,
    y_ptr,
    N,
    D,
    r,
    g_row,
    g_col,
    HAS_GATE: tl.constexpr,
    SCALED: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_D)
    rok = rows < N
    mask = rok[:, None] & (cols < D)[None, :]
    offs = rows.to(tl.int64)[:, None] * D + cols[None, :]
    o = tl.load(o_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_GATE:
        s = tl.load(seg_ptr + rows, mask=rok, other=0).to(tl.int64)
        o = o * tl.load(
            gate_ptr + s[:, None] * g_row + cols[None, :] * g_col, mask=mask, other=0.0
        ).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if SCALED:
        x = x * r
    tl.store(y_ptr + offs, (x + o).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_kernel(
    g_ptr,
    o_ptr,
    gate_ptr,
    cu_ptr,
    dx_ptr,
    do_ptr,
    part_ptr,
    N,
    D,
    B,
    CHUNK,
    r,
    g_row,
    g_col,
    NEED_DX: tl.constexpr,
    NEED_GATE: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    t = tl.program_id(0)
    b, start, stop = tile_range(t, cu_ptr, N, B, CHUNK, True, BLOCK_B)
    cols = tl.arange(0, BLOCK_D)
    cok = cols < D
    goff = b.to(tl.int64) * g_row + cols * g_col
    gate = tl.load(gate_ptr + goff, mask=cok, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK_D], tl.float32)
    for r0 in range(start, stop, ROWS):
        rows = r0 + tl.arange(0, ROWS)
        mask = (rows < stop)[:, None] & cok[None, :]
        offs = rows.to(tl.int64)[:, None] * D + cols[None, :]
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            do_ptr + offs, (g * gate[None, :]).to(do_ptr.dtype.element_ty), mask=mask
        )
        if NEED_DX:
            tl.store(dx_ptr + offs, (g * r).to(dx_ptr.dtype.element_ty), mask=mask)
        if NEED_GATE:
            o = tl.load(o_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(g * o, axis=0)
    if NEED_GATE:
        tl.store(part_ptr + t.to(tl.int64) * D + cols, acc, mask=cok)


_fwd = Launcher(_fwd_kernel)
_bwd = Launcher(_bwd_kernel)


def _fwd_launch(cfg, x, o, gate, seg, y, r):
    n, d = x.shape
    _fwd[(cdiv(n, cfg["ROWS"]),)](
        x, o, gate if gate is not None else x, seg if seg is not None else x, y, n, d, r,
        *strides2(gate), HAS_GATE=gate is not None, SCALED=r != 1.0, ROWS=cfg["ROWS"],
        BLOCK_D=triton.next_power_of_2(d), num_warps=cfg["warps"],
    )  # fmt: skip


@triton_op("kohakumoss::triton_gated_residual_fwd", mutates_args=())
def _fwd_op(
    x: Tensor, o: Tensor, gate: Tensor | None, seg: Tensor | None, r: float
) -> Tensor:
    return residual_forward(x, o, gate, seg, r)


def residual_forward(x, o, gate, seg, r):
    """x * r + o * gate[seg] for x, o [N, D] (no autograd)."""
    y = torch.empty_like(x)
    key = f"d{x.shape[1]}-{x.dtype}-{gate is not None}"
    block = triton.next_power_of_2(x.shape[1])
    cfg = select(
        "triton_gated_residual_fwd",
        x.device,
        key,
        lambda: row_candidates(block),
        lambda c: lambda: _fwd_launch(c, x, o, gate, seg, y, r),
    )
    _fwd_launch(cfg, x, o, gate, seg, y, r)
    return y


def _bwd_launch(cfg, g, o, gate, seg, b, r, dx, do, need_gate):
    n, d = g.shape
    per_sm = cfg["per_sm"] if need_gate else None
    plan = SegPlan(seg, n, b, cfg["ROWS"], g.device, per_sm)
    part = torch.empty(
        plan.tiles if need_gate else 1, 1, d, device=g.device, dtype=torch.float32
    )
    _bwd[(plan.tiles,)](
        g, o, gate, plan.cu, dx, do, part, n, d, plan.b, plan.chunk, r, *strides2(gate),
        NEED_DX=r != 1.0, NEED_GATE=need_gate, ROWS=cfg["ROWS"],
        BLOCK_D=triton.next_power_of_2(d), BLOCK_B=plan.block_b, num_warps=cfg["warps"],
    )  # fmt: skip
    return plan, part


def residual_backward(g, o, gate, seg, r, need_gate):
    """(d_x or empty when r == 1, d_o in o's dtype, d_gate fp32 or empty); o may be
    empty (of its dtype) when d_gate is not needed (no autograd)."""
    n, d = g.shape
    b = gate.shape[0]
    do = torch.empty_like(g, dtype=o.dtype)
    dx = torch.empty_like(g) if r != 1.0 else g.new_empty(0)
    key = f"d{d}-{g.dtype}-{need_gate}-n{pow2(n, 1024)}"
    block = triton.next_power_of_2(d)
    cands = reduce_candidates(block) if need_gate else row_candidates(block)
    cfg = select(
        "triton_gated_residual_bwd",
        g.device,
        key,
        lambda: cands,
        lambda c: lambda: _bwd_launch(c, g, o, gate, seg, b, r, dx, do, need_gate),
        shortlist=len(cands),
        in_backward=True,
    )
    plan, part = _bwd_launch(cfg, g, o, gate, seg, b, r, dx, do, need_gate)
    dgate = plan.reduce(part)[0] if need_gate else g.new_empty(0)
    return dx, do, dgate


@triton_op("kohakumoss::triton_gated_residual_bwd", mutates_args=())
def _bwd_op(
    g: Tensor, o: Tensor, gate: Tensor, seg: Tensor, r: float, need_gate: bool
) -> tuple[Tensor, Tensor, Tensor]:
    """(d_x or empty when r == 1, d_o, d_gate fp32 or empty)."""
    return residual_backward(g, o, gate, seg, r, need_gate)


def _setup(ctx, inputs, output):
    x, o, gate, seg, r = inputs
    gate_grad = gate is not None and ctx.needs_input_grad[2]
    ctx.save_for_backward(o if gate_grad else o.new_empty(0), gate, seg)
    ctx.r = r
    ctx.x_dtype = x.dtype
    ctx.o_dtype = o.dtype


def _backward_with(bwd):
    """The autograd backward over `bwd` (the backward op, or its body in eager)."""

    def backward(ctx, gy):
        o, gate, seg = ctx.saved_tensors
        r = ctx.r
        need = ctx.needs_input_grad
        if gate is None:
            dx = gy if r == 1.0 else (gy.float() * r).to(ctx.x_dtype)
            return dx, gy.to(ctx.o_dtype), None, None, None
        dx, do, dgate = bwd(gy.contiguous(), o, gate, seg, r, need[2])
        dx = gy if r == 1.0 else dx
        return dx, do, dgate.to(gate.dtype) if need[2] else None, None, None

    return backward


_fwd_op.register_autograd(_backward_with(_bwd_op), setup_context=_setup)
_Eager = eager_function(
    "GatedResidual", residual_forward, _setup, _backward_with(residual_backward)
)


def gated_residual(
    x: Tensor, o: Tensor, gate: Tensor | None, seg: Tensor | None, r: float
) -> Tensor:
    """`x * r + o * gate[seg]`, as the eager reference computes it; x, o [N, D], gate
    [B, D] per sample (any strides)."""
    shape = x.shape
    d = shape[-1]
    fwd = _fwd_op if torch.compiler.is_compiling() else _Eager.apply
    y = fwd(
        x.reshape(-1, d).contiguous(),
        o.reshape(-1, d).contiguous(),
        gate,
        None if seg is None else seg.contiguous(),
        float(r),
    )
    return y.view(shape)


register(
    "gated_residual",
    "triton",
    kernel_or_compile("triton", "gated_residual", gated_residual, shape_key),
    requires(kinds=("cuda", "xpu"), min_sm=70, dtypes=DTYPES),
)
