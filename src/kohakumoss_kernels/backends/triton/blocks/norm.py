"""Triton RMSNorm and RMSNorm + per-sample modulation, forward and backward.

Forward: one row tile `[ROWS, next_pow2(D)]` per program, fp32 math, one rounding;
`rstd [N]` fp32 is kept for the backward. Backward: one program per segment tile writes
`dx` and per-tile fp32 column partials of `sum(g * xhat)` and
`sum(g)`; a segment sum and an epilogue give d_scale = w * S_gx, d_shift = S_g and
d_weight = sum_b (1 + scale_b) * S_gx without atomics. See docs/internals/fp16.md.
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
    num_sms,
    opt,
    reduce_candidates,
    row_candidates,
    segment_tiles,
    strides2,
    tile_range,
)
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.buckets import cdiv, pow2
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
    sh_row,
    sh_col,
    sc_row,
    sc_col,
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
    if HAS_SHIFT or HAS_SCALE:
        s = tl.load(seg_ptr + rows, mask=rok, other=0).to(tl.int64)
    ms = tl.div_rn(tl.sum(x * x, axis=1), D * 1.0)
    rstd = tl.div_rn(1.0, tl.sqrt_rn(ms + eps.to(tl.float32)))
    y = x * rstd[:, None]
    if HAS_W:
        y = y * tl.load(w_ptr + cols, mask=cok, other=0.0).to(tl.float32)[None, :]
    if HAS_SCALE:
        soff = s[:, None] * sc_row + cols[None, :] * sc_col
        sc = tl.load(scale_ptr + soff, mask=mask, other=0.0).to(tl.float32)
        y = y * (1.0 + sc)
    if HAS_SHIFT:
        hoff = s[:, None] * sh_row + cols[None, :] * sh_col
        y = y + tl.load(shift_ptr + hoff, mask=mask, other=0.0).to(tl.float32)
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
    sc_row,
    sc_col,
    HAS_W: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SEG: tl.constexpr,
    NEED_P: tl.constexpr,
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
        soff = b.to(tl.int64) * sc_row + cols * sc_col
        sc = tl.load(scale_ptr + soff, mask=cok, other=0.0).to(tl.float32)
        m = m * (1.0 + sc)
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
        xhat = x * rs[:, None]
        gm = g * m[None, :]
        c = tl.div_rn(tl.sum(gm * xhat, axis=1), D * 1.0)
        dx = rs[:, None] * (gm - xhat * c[:, None])
        tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)
        if NEED_P:
            acc_gx += tl.sum(g * xhat, axis=0)
            acc_g += tl.sum(g, axis=0)
    if NEED_P:
        base = part_ptr + t.to(tl.int64) * 2 * D + cols
        tl.store(base, acc_gx, mask=cok)
        tl.store(base + D, acc_g, mask=cok)


@triton.jit
def _tile_sums(
    part_ptr, beg, end, cols, ok, D, SLOT: tl.constexpr, BT: tl.constexpr,
    BD: tl.constexpr,
):  # fmt: skip
    """fp32 sum of slot SLOT of the tile partials [T, 2, D] over tiles [beg, end)."""
    acc = tl.zeros([BD], tl.float32)
    for t0 in range(beg, end, BT):
        ts = t0 + tl.arange(0, BT)
        m = (ts < end)[:, None] & ok[None, :]
        src = part_ptr + (ts.to(tl.int64)[:, None] * 2 + SLOT) * D + cols[None, :]
        acc += tl.sum(tl.load(src, mask=m, other=0.0), axis=0)
    return acc


@triton.jit
def _grads_kernel(
    part_ptr,
    cu_ptr,
    w_ptr,
    scale_ptr,
    dw_ptr,
    dshift_ptr,
    dscale_ptr,
    B,
    D,
    CHUNK,
    T,
    sc_row,
    sc_col,
    HAS_W: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SEG: tl.constexpr,
    NEED_W: tl.constexpr,
    NEED_SHIFT: tl.constexpr,
    NEED_SCALE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """d_weight, d_shift, d_scale (fp32) straight from the tile partials: per sample,
    S_gx and S_g are fp32 sums over its tiles; d_weight sums over samples."""
    cols = tl.program_id(0) * BD + tl.arange(0, BD)
    ok = cols < D
    w = tl.full([BD], 1.0, tl.float32)
    if HAS_W:
        w = tl.load(w_ptr + cols, mask=ok, other=0.0).to(tl.float32)
    acc = tl.zeros([BD], tl.float32)
    for b in range(B):
        beg, end = segment_tiles(b, cu_ptr, B, CHUNK, T, HAS_SEG, BLOCK_B)
        gx = _tile_sums(part_ptr, beg, end, cols, ok, D, 0, BT, BD)
        if NEED_SHIFT:
            g = _tile_sums(part_ptr, beg, end, cols, ok, D, 1, BT, BD)
            tl.store(dshift_ptr + b * D + cols, g, mask=ok)
        if NEED_SCALE:
            tl.store(dscale_ptr + b * D + cols, w * gx, mask=ok)
        if NEED_W:
            if HAS_SCALE:
                soff = b.to(tl.int64) * sc_row + cols * sc_col
                sc = tl.load(scale_ptr + soff, mask=ok, other=0.0).to(tl.float32)
                gx = gx * (1.0 + sc)
            acc += gx
    if NEED_W:
        tl.store(dw_ptr + cols, acc, mask=ok)


_fwd = Launcher(_fwd_kernel)
_bwd = Launcher(_bwd_kernel)
_grads = Launcher(_grads_kernel)


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
        *strides2(shift),
        *strides2(scale),
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
        "triton_rms_norm_fwd",
        x.device,
        key,
        lambda: row_candidates(block),
        lambda c: lambda: _fwd_launch(c, x, w, shift, scale, seg, y, rstd, eps),
    )
    _fwd_launch(cfg, x, w, shift, scale, seg, y, rstd, eps)
    return y, rstd


@triton_op("kohakumoss::triton_rms_norm_fwd", mutates_args=())
def _fwd_op(
    x: Tensor,
    weight: Tensor | None,
    shift: Tensor | None,
    scale: Tensor | None,
    seg: Tensor | None,
    eps: float,
) -> tuple[Tensor, Tensor]:
    return _fwd_impl(x, weight, shift, scale, seg, eps)


def _bwd_launch(cfg, g, x, rstd, w, scale, seg, b, dx, need_p):
    n, d = x.shape
    rows = cfg["ROWS"]
    plan = SegPlan(
        seg if need_p or scale is not None else None,
        n,
        b,
        rows,
        x.device,
        cfg["per_sm"] if need_p else None,
    )
    part = torch.empty(
        plan.tiles if need_p else 1, 2, d, device=x.device, dtype=torch.float32
    )
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
        *strides2(scale),
        HAS_W=w is not None,
        HAS_SCALE=scale is not None,
        HAS_SEG=plan.has_seg,
        NEED_P=need_p,
        ROWS=rows,
        BLOCK_D=triton.next_power_of_2(d),
        BLOCK_B=plan.block_b,
        num_warps=cfg["warps"],
    )
    return plan, part


def _grads_launch(plan, part, w, scale, d, need_w, need_shift, need_scale):
    f32 = dict(device=part.device, dtype=torch.float32)
    b = plan.b
    dw = torch.empty(d if need_w else 0, **f32)
    dshift = torch.empty((b, d) if need_shift else 0, **f32)
    dscale = torch.empty((b, d) if need_scale else 0, **f32)
    spread = triton.next_power_of_2(cdiv(d, 2 * num_sms(part.device)))
    bd = max(32, min(256, spread))
    _grads[(cdiv(d, bd),)](
        part,
        plan.cu,
        w if w is not None else part,
        scale if scale is not None else part,
        dw,
        dshift,
        dscale,
        b,
        d,
        plan.chunk,
        plan.tiles,
        *strides2(scale),
        HAS_W=w is not None,
        HAS_SCALE=scale is not None,
        HAS_SEG=plan.has_seg,
        NEED_W=need_w,
        NEED_SHIFT=need_shift,
        NEED_SCALE=need_scale,
        BLOCK_B=plan.block_b,
        BT=max(8, 2048 // bd),
        BD=bd,
        num_warps=2,
    )
    return dw, dshift, dscale


def _bwd_impl(
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
    """(dx, dweight, dshift fp32, dscale fp32); gradients not needed come back empty."""
    n, d = x.shape
    dx = torch.empty_like(x)
    block = triton.next_power_of_2(d)
    need_p = need_w or need_shift or need_scale
    key = f"d{d}-{x.dtype}-{weight is not None}{scale is not None}{seg is not None}"
    key = f"{key}-{need_p}-n{pow2(n, 1024)}"
    args = (g, x, rstd, weight, scale, seg, batch, dx, need_p)
    cands = reduce_candidates(block) if need_p else row_candidates(block)
    cfg = select(
        "triton_rms_norm_bwd",
        x.device,
        key,
        lambda: cands,
        lambda c: lambda: _bwd_launch(c, *args),
        shortlist=len(cands),
        in_backward=True,
    )
    plan, part = _bwd_launch(cfg, *args)
    if not need_p:
        return dx, x.new_empty(0), x.new_empty(0), x.new_empty(0)
    dw, dshift, dscale = _grads_launch(
        plan, part, weight, scale, d, need_w, need_shift, need_scale
    )
    return dx, dw.to(weight.dtype) if need_w else dw, dshift, dscale


_bwd_op = triton_op("kohakumoss::triton_rms_norm_bwd", mutates_args=())(_bwd_impl)


def _setup(ctx, inputs, output):
    x, weight, shift, scale, seg, eps = inputs
    ctx.save_for_backward(x, output[1], weight, scale, seg)
    ctx.shift_dtype = None if shift is None else shift.dtype
    ctx.batch = (
        shift.shape[0]
        if shift is not None
        else scale.shape[0] if scale is not None else 1
    )


def _backward_with(bwd):
    """The autograd backward over `bwd` (the backward op, or its body in eager)."""

    def backward(ctx, gy, _grstd):
        x, rstd, weight, scale, seg = ctx.saved_tensors
        need = ctx.needs_input_grad
        dx, dw, dshift, dscale = bwd(
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

    return backward


_fwd_op.register_autograd(_backward_with(_bwd_op), setup_context=_setup)
_Eager = eager_function(
    "RmsNormMod", _fwd_impl, _setup, _backward_with(_bwd_impl), nondiff=(1,)
)


def rms_norm_mod(
    x: Tensor,
    weight: Tensor | None,
    shift: Tensor | None,
    scale: Tensor | None,
    seg: Tensor | None,
    eps: float,
) -> Tensor:
    """`rms_norm(x) * (1 + scale[seg]) + shift[seg]` over the last dim, as the eager
    reference computes it; shift / scale [B, D] may be strided views (read through
    their strides); `seg` is read only with a shift or scale."""
    shape = x.shape
    if shift is None and scale is None:
        seg = None
    fwd = _fwd_op if torch.compiler.is_compiling() else _Eager.apply
    y, _ = fwd(
        x.reshape(-1, shape[-1]).contiguous(),
        None if weight is None else weight.contiguous(),
        shift,
        scale,
        None if seg is None else seg.contiguous(),
        eps,
    )
    return y.view(shape)


def rms_norm(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """`x / rms(x) * weight` over the last dim, as the eager reference computes it."""
    return rms_norm_mod(x, weight, None, None, None, eps)


_SUPPORTS = requires(kinds=("cuda", "xpu"), min_sm=70, dtypes=DTYPES)
register(
    "rms_norm",
    "triton",
    kernel_or_compile("triton", "rms_norm", rms_norm, shape_key),
    _SUPPORTS,
)
register(
    "rms_norm_mod",
    "triton",
    kernel_or_compile("triton", "rms_norm_mod", rms_norm_mod, shape_key),
    _SUPPORTS,
)
