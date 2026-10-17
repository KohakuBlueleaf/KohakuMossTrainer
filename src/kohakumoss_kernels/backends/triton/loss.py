"""Triton token MSE: sum over tokens of weight[seg] * mask * mean_c((pred - target)^2).

Forward computes in double-float (fp32 pairs) and writes one pair partial per program,
pair-summed afterwards (deterministic) and rounded once. The
backward writes d_pred (and d_target = -d_pred when asked); `weight` and `mask` are
data, not trained, and get no gradient.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op

from kohakumoss_kernels.backends.compile.race import raced, shape_key
from kohakumoss_kernels.backends.triton._blockops import (
    df_div,
    df_mul,
    df_sum,
    total,
    two_prod,
    two_sum,
)
from kohakumoss_kernels.backends.triton._common import Launcher, cdiv
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.registry import register

__all__ = ["token_mse"]

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
ROWS = 16


@triton.jit
def _fwd_kernel(
    p_ptr,
    t_ptr,
    seg_ptr,
    w_ptr,
    m_ptr,
    part_ptr,
    N,
    C,
    HAS_MASK: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_C)
    rok = rows < N
    mask = rok[:, None] & (cols < C)[None, :]
    offs = rows.to(tl.int64)[:, None] * C + cols[None, :]
    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(t_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dh, dl = two_sum(p, -t)
    sh, sl = df_mul(dh, dl, dh, dl)
    eh, el = df_sum(sh, sl, 1)
    eh, el = df_div(eh, el, C * 1.0)
    s = tl.load(seg_ptr + rows, mask=rok, other=0)
    w = tl.load(w_ptr + s, mask=rok, other=0.0).to(tl.float32)
    wl = 0.0 * w
    if HAS_MASK:
        w, wl = two_prod(w, tl.load(m_ptr + rows, mask=rok, other=0.0).to(tl.float32))
    eh, el = df_mul(eh, el, w, wl)
    eh, el = df_sum(eh, el, 0)
    tl.store(part_ptr + pid * 2, eh)
    tl.store(part_ptr + pid * 2 + 1, el)


@triton.jit
def _bwd_kernel(
    gs_ptr,
    p_ptr,
    t_ptr,
    seg_ptr,
    w_ptr,
    m_ptr,
    dp_ptr,
    dt_ptr,
    N,
    C,
    HAS_MASK: tl.constexpr,
    NEED_T: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_C)
    rok = rows < N
    mask = rok[:, None] & (cols < C)[None, :]
    offs = rows.to(tl.int64)[:, None] * C + cols[None, :]
    d = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32) - tl.load(
        t_ptr + offs, mask=mask, other=0.0
    ).to(tl.float32)
    s = tl.load(seg_ptr + rows, mask=rok, other=0)
    w = tl.load(w_ptr + s, mask=rok, other=0.0).to(tl.float32)
    if HAS_MASK:
        w = w * tl.load(m_ptr + rows, mask=rok, other=0.0).to(tl.float32)
    coef = tl.load(gs_ptr).to(tl.float32) * tl.div_rn(2.0, C * 1.0) * w
    dp = d * coef[:, None]
    tl.store(dp_ptr + offs, dp.to(dp_ptr.dtype.element_ty), mask=mask)
    if NEED_T:
        tl.store(dt_ptr + offs, (-dp).to(dt_ptr.dtype.element_ty), mask=mask)


_fwd = Launcher(_fwd_kernel)
_bwd = Launcher(_bwd_kernel)


@triton_op("kohakumoss::triton_token_mse_fwd", mutates_args=())
def _fwd_op(
    pred: Tensor, target: Tensor, seg: Tensor, weight: Tensor, mask: Tensor | None
) -> Tensor:
    n, c = pred.shape
    progs = cdiv(n, ROWS)
    part = torch.empty(progs, 1, 2, 1, device=pred.device, dtype=torch.float32)
    _fwd[(progs,)](
        pred, target, seg, weight, mask if mask is not None else weight, part, n, c,
        HAS_MASK=mask is not None, ROWS=ROWS, BLOCK_C=triton.next_power_of_2(c),
        num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    return total(part).reshape(())


@triton_op("kohakumoss::triton_token_mse_bwd", mutates_args=())
def _bwd_op(
    gs: Tensor,
    pred: Tensor,
    target: Tensor,
    seg: Tensor,
    weight: Tensor,
    mask: Tensor | None,
    need_target: bool,
) -> tuple[Tensor, Tensor]:
    n, c = pred.shape
    dp = torch.empty_like(pred)
    dt = torch.empty_like(target) if need_target else pred.new_empty(0)
    _bwd[(cdiv(n, ROWS),)](
        gs, pred, target, seg, weight, mask if mask is not None else weight, dp,
        dt if need_target else dp, n, c, HAS_MASK=mask is not None, NEED_T=need_target,
        ROWS=ROWS, BLOCK_C=triton.next_power_of_2(c), num_warps=4,
    )  # fmt: skip
    return dp, dt


def _setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, gs):
    pred, target, seg, weight, mask = ctx.saved_tensors
    need = ctx.needs_input_grad
    dp, dt = _bwd_op(
        gs.float().reshape(1).contiguous(), pred, target, seg, weight, mask, need[1]
    )
    return dp if need[0] else None, dt if need[1] else None, None, None, None


_fwd_op.register_autograd(_backward, setup_context=_setup)


def token_mse(
    pred: Tensor, target: Tensor, seg: Tensor, weight: Tensor, mask: Tensor | None
) -> Tensor:
    """fp32 sum over tokens of weight[seg] * mask * mean_c((pred - target)^2); the 1-D
    weight / mask / seg are read densely (a strided one is copied)."""
    c = pred.shape[-1]
    return _fwd_op(
        pred.reshape(-1, c).contiguous(),
        target.reshape(-1, c).contiguous(),
        seg.contiguous(),
        weight.contiguous(),
        None if mask is None else mask.reshape(-1).contiguous(),
    )


register(
    "token_mse",
    "triton",
    raced("triton", "token_mse", token_mse, shape_key),
    requires(kinds=("cuda", "xpu"), min_sm=70, dtypes=DTYPES),
)
