"""Triton RMSNorm and RMSNorm + per-sample modulation, forward and backward.

Forward: one row tile `[ROWS, next_pow2(D)]` per program, fp32 math, one rounding;
`rstd [N]` fp32 is kept for the backward. Backward: one program per segment tile
writes `dx` and per-tile column partials of `sum(g * xhat)` and `sum(g)`; a second
launch sums them per sample, giving d_scale = w * S_gx, d_shift = S_g and
d_weight = sum_b (1 + scale_b) * S_gx without atomics.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op

from kohakumoss_kernels.backends.compile.race import raced, shape_key
from kohakumoss_kernels.backends.triton._blockops import (
    SegPlan,
    opt,
    row_candidates,
    tile_range,
)
from kohakumoss_kernels.backends.triton._common import Launcher, cdiv
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.tuner import select

__all__ = ["rms_norm", "rms_norm_mod"]

DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _fwd_kernel(
    x_ptr,
    w_ptr,
    shift_ptr,
    scale_ptr,
    seg_ptr,
    y_ptr,
    rstd_ptr,
    N,
    D,
    eps,
    HAS_W: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_D)
    rok = rows < N
    cok = cols < D
    mask = rok[:, None] & cok[None, :]
    offs = rows.to(tl.int64)[:, None] * D + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.div_rn(tl.sum(x * x, axis=1), D * 1.0)
    rstd = tl.div_rn(1.0, tl.sqrt_rn(ms + eps.to(tl.float32)))
    y = x * rstd[:, None]
    if HAS_W:
        y = y * tl.load(w_ptr + cols, mask=cok, other=0.0).to(tl.float32)[None, :]
    if HAS_SHIFT or HAS_SCALE:
        s = tl.load(seg_ptr + rows, mask=rok, other=0).to(tl.int64)
        moff = s[:, None] * D + cols[None, :]
        if HAS_SCALE:
            sc = tl.load(scale_ptr + moff, mask=mask, other=0.0).to(tl.float32)
            y = y * (1.0 + sc)
        if HAS_SHIFT:
            y = y + tl.load(shift_ptr + moff, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + rows, rstd, mask=rok)


@triton.jit
def _bwd_kernel(
    g_ptr,
    x_ptr,
    rstd_ptr,
    w_ptr,
    scale_ptr,
    cu_ptr,
    dx_ptr,
    part_ptr,
    N,
    D,
    B,
    CHUNK,
    HAS_W: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SEG: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    t = tl.program_id(0)
    b, start, stop = tile_range(t, cu_ptr, N, B, CHUNK, HAS_SEG, BLOCK_B)
    cols = tl.arange(0, BLOCK_D)
    cok = cols < D
    m = tl.full([BLOCK_D], 1.0, tl.float32)
    if HAS_W:
        m = tl.load(w_ptr + cols, mask=cok, other=0.0).to(tl.float32)
    if HAS_SCALE:
        sc = tl.load(scale_ptr + b.to(tl.int64) * D + cols, mask=cok, other=0.0)
        m = m * (1.0 + sc.to(tl.float32))
    acc_gx = tl.zeros([BLOCK_D], tl.float32)
    acc_g = tl.zeros([BLOCK_D], tl.float32)
    for r0 in range(start, stop, ROWS):
        rows = r0 + tl.arange(0, ROWS)
        rok = rows < stop
        mask = rok[:, None] & cok[None, :]
        offs = rows.to(tl.int64)[:, None] * D + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        rs = tl.load(rstd_ptr + rows, mask=rok, other=0.0)
        xh = x * rs[:, None]
        gm = g * m[None, :]
        c = tl.div_rn(tl.sum(gm * xh, axis=1), D * 1.0)
        dx = rs[:, None] * (gm - xh * c[:, None])
        tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)
        acc_gx += tl.sum(g * xh, axis=0)
        acc_g += tl.sum(g, axis=0)
    base = t.to(tl.int64) * 2 * D
    tl.store(part_ptr + base + cols, acc_gx, mask=cok)
    tl.store(part_ptr + base + D + cols, acc_g, mask=cok)


_fwd = Launcher(_fwd_kernel)
_bwd = Launcher(_bwd_kernel)


def _fwd_launch(cfg, x, w, shift, scale, seg, y, rstd, eps):
    n, d = x.shape
    rows = cfg["ROWS"]
    _fwd[(cdiv(n, rows),)](
        x,
        w if w is not None else x,
        shift if shift is not None else x,
        scale if scale is not None else x,
        seg if seg is not None else x,
        y,
        rstd,
        n,
        d,
        eps,
        HAS_W=w is not None,
        HAS_SHIFT=shift is not None,
        HAS_SCALE=scale is not None,
        ROWS=rows,
        BLOCK_D=triton.next_power_of_2(d),
        num_warps=cfg["warps"],
    )


def _fwd_impl(x, w, shift, scale, seg, eps):
    n, d = x.shape
    y = torch.empty_like(x)
    rstd = torch.empty(n, device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(d)
    key = f"d{d}-{x.dtype}-{w is not None}{shift is not None}{scale is not None}"
    cfg = select(
        "f6_rms_norm_fwd",
        x.device,
        key,
        lambda: row_candidates(block),
        lambda c: lambda: _fwd_launch(c, x, w, shift, scale, seg, y, rstd, eps),
    )
    _fwd_launch(cfg, x, w, shift, scale, seg, y, rstd, eps)
    return y, rstd


@triton_op("kohakumoss::f6_rms_norm_fwd", mutates_args=())
def _fwd_op(
    x: Tensor,
    weight: Tensor | None,
    shift: Tensor | None,
    scale: Tensor | None,
    seg: Tensor | None,
    eps: float,
) -> tuple[Tensor, Tensor]:
    return _fwd_impl(x, weight, shift, scale, seg, eps)


def _bwd_launch(cfg, g, x, rstd, w, scale, seg, b, dx):
    n, d = x.shape
    rows = cfg["ROWS"]
    plan = SegPlan(seg, n, b, rows, x.device)
    part = torch.empty(plan.tiles, 2, d, device=x.device, dtype=torch.float32)
    _bwd[(plan.tiles,)](
        g,
        x,
        rstd,
        w if w is not None else x,
        scale if scale is not None else x,
        plan.cu,
        dx,
        part,
        n,
        d,
        plan.b,
        plan.chunk,
        HAS_W=w is not None,
        HAS_SCALE=scale is not None,
        HAS_SEG=plan.has_seg,
        ROWS=rows,
        BLOCK_D=triton.next_power_of_2(d),
        BLOCK_B=plan.block_b,
        num_warps=cfg["warps"],
    )
    return plan, part


@triton_op("kohakumoss::f6_rms_norm_bwd", mutates_args=())
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
    """(dx, dweight, dshift, dscale); gradients not needed come back empty."""
    n, d = x.shape
    dx = torch.empty_like(x)
    block = triton.next_power_of_2(d)
    key = f"d{d}-{x.dtype}-{weight is not None}{scale is not None}{seg is not None}"
    cfg = select(
        "f6_rms_norm_bwd",
        x.device,
        key,
        lambda: row_candidates(block),
        lambda c: lambda: _bwd_launch(c, g, x, rstd, weight, scale, seg, batch, dx),
    )
    plan, part = _bwd_launch(cfg, g, x, rstd, weight, scale, seg, batch, dx)
    dw, dshift, dscale = x.new_empty(0), x.new_empty(0), x.new_empty(0)
    if not (need_w or need_shift or need_scale):
        return dx, dw, dshift, dscale
    sums = plan.reduce(part, 2, d)
    s_gx, s_g = sums[0], sums[1]
    if need_w:
        coef = s_gx if scale is None else s_gx * (1.0 + scale.float())
        dw = coef.sum(0).to(weight.dtype)
    if need_shift:
        dshift = s_g.clone()
    if need_scale:
        dscale = s_gx.clone() if weight is None else s_gx * weight.float()
    return dx, dw, dshift, dscale


def _setup(ctx, inputs, output):
    x, weight, shift, scale, seg, _ = inputs
    ctx.save_for_backward(x, output[1], weight, scale, seg)
    ctx.shift_dtype = None if shift is None else shift.dtype
    ctx.batch = (
        shift.shape[0]
        if shift is not None
        else scale.shape[0] if scale is not None else 1
    )


def _backward(ctx, gy, _grstd):
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
    )
    dshift = opt(dshift)
    dscale = opt(dscale)
    return (
        dx,
        opt(dw),
        None if dshift is None else dshift.to(ctx.shift_dtype),
        None if dscale is None else dscale.to(scale.dtype),
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
    """`rms_norm(x) * (1 + scale[seg]) + shift[seg]` over the last dim (eager spec)."""
    shape = x.shape
    y, _ = _fwd_op(
        x.reshape(-1, shape[-1]).contiguous(), weight, shift, scale, seg, eps
    )
    return y.view(shape)


def rms_norm(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """`x / rms(x) * weight` over the last dim (eager spec)."""
    return rms_norm_mod(x, weight, None, None, None, eps)


_SUPPORTS = requires(kinds=("cuda", "xpu"), min_sm=70, dtypes=DTYPES)
register(
    "rms_norm", "triton", raced("triton", "rms_norm", rms_norm, shape_key), _SUPPORTS
)
register(
    "rms_norm_mod",
    "triton",
    raced("triton", "rms_norm_mod", rms_norm_mod, shape_key),
    _SUPPORTS,
)
