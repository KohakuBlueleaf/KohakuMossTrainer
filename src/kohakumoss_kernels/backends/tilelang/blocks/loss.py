"""TileLang token MSE: sum over tokens of weight[seg] * mask * mean_c((pred - target)^2).

Forward computes in fp32, one row per thread, and writes one partial per row, summed
afterwards in a fixed order; `weight` and `mask` are data and get no gradient.
"""

import tilelang.language as T
import torch
from torch import Tensor
from torch.library import custom_op

from kohakumoss_kernels.backends.compile.choice import kernel_or_compile, shape_key
from kohakumoss_kernels.backends.eager import loss as eager_loss
from kohakumoss_kernels.backends.tilelang.blocks._blockops import (
    DTYPE,
    F32,
    FP16_SUPPORTS,
    I32,
    CompileError,
    compiled,
    segment_ids,
    total,
)
from kohakumoss_kernels.registry import register
from kohakumoss_kernels.tuning.buckets import cdiv

__all__ = ["token_mse"]

FWD_ROWS = 128
BWD_ROWS, BWD_THREADS = 16, 128


def _fwd_prim(c, dtype, tdtype):
    """One row per thread, fp32 over the row's channels; per-row partials of
    weight * mask * mean((p - t)^2) (`t` in its own dtype)."""
    n = T.dynamic("n")
    b = T.dynamic("b")
    pp = T.dynamic("p")

    @T.prim_func
    def main(
        p: T.Tensor((n, c), dtype),
        t: T.Tensor((n, c), tdtype),
        seg: T.Tensor((n,), I32),
        w: T.Tensor((b,), F32),
        m: T.Tensor((n,), F32),
        part: T.Tensor((pp, FWD_ROWS, 1, 1), F32),
    ):
        with T.Kernel(T.ceildiv(n, FWD_ROWS), threads=FWD_ROWS) as bx:
            e = T.alloc_fragment((FWD_ROWS,), F32)
            T.clear(e)
            for i in T.Parallel(FWD_ROWS):
                rr = T.min(bx * FWD_ROWS + i, n - 1)
                for j in T.serial(c):
                    dv = T.cast(p[rr, j], F32) - T.cast(t[rr, j], F32)
                    e[i] = e[i] + dv * dv
            for i in T.Parallel(FWD_ROWS):
                row = bx * FWD_ROWS + i
                rr = T.min(row, n - 1)
                v = e[i] / T.float32(c) * (w[seg[rr]] * m[rr])
                part[bx, i, 0, 0] = T.if_then_else(row < n, v, 0.0)

    return main


def _bwd_prim(c, dtype, tdtype):
    n = T.dynamic("n")
    b = T.dynamic("b")

    @T.prim_func
    def main(
        gs: T.Tensor((1,), F32),
        p: T.Tensor((n, c), dtype),
        t: T.Tensor((n, c), tdtype),
        seg: T.Tensor((n,), I32),
        w: T.Tensor((b,), F32),
        m: T.Tensor((n,), F32),
        dp: T.Tensor((n, c), dtype),
    ):
        with T.Kernel(T.ceildiv(n, BWD_ROWS), threads=BWD_THREADS) as bx:
            for i, j in T.Parallel(BWD_ROWS, c):
                row = bx * BWD_ROWS + i
                rr = T.min(row, n - 1)
                coef = gs[0] * 2.0 / c * w[seg[rr]] * m[rr]
                d = T.cast(p[rr, j], F32) - T.cast(t[rr, j], F32)
                if row < n:
                    dp[row, j] = T.cast(d * coef, dtype)

    return main


def _mask(mask, n, device):
    return (
        torch.ones(n, device=device)
        if mask is None
        else mask.to(torch.float32).contiguous()
    )


@custom_op("kohakumoss::tilelang_token_mse_fwd", mutates_args=())
def _fwd_op(
    pred: Tensor, target: Tensor, seg: Tensor, weight: Tensor, mask: Tensor | None
) -> Tensor:
    n, c = pred.shape
    part = torch.empty(
        cdiv(n, FWD_ROWS), FWD_ROWS, 1, 1, device=pred.device, dtype=torch.float32
    )
    kernel = compiled(_fwd_prim, pred.device, c, DTYPE[pred.dtype], DTYPE[target.dtype])
    kernel(
        pred,
        target,
        segment_ids(seg, n, pred.device),
        weight.float().contiguous(),
        _mask(mask, n, pred.device),
        part,
    )
    return total(part).reshape(())


@_fwd_op.register_fake
def _(pred, target, seg, weight, mask):
    return pred.new_empty((), dtype=torch.float32)


@custom_op("kohakumoss::tilelang_token_mse_bwd", mutates_args=())
def _bwd_op(
    gs: Tensor,
    pred: Tensor,
    target: Tensor,
    seg: Tensor,
    weight: Tensor,
    mask: Tensor | None,
) -> Tensor:
    n, c = pred.shape
    dp = torch.empty_like(pred)
    kernel = compiled(_bwd_prim, pred.device, c, DTYPE[pred.dtype], DTYPE[target.dtype])
    kernel(
        gs,
        pred,
        target,
        segment_ids(seg, n, pred.device),
        weight.float().contiguous(),
        _mask(mask, n, pred.device),
        dp,
    )
    return dp


@_bwd_op.register_fake
def _(gs, pred, target, seg, weight, mask):
    return torch.empty_like(pred)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, gs):
    pred, target, seg, weight, mask = ctx.saved_tensors
    need = ctx.needs_input_grad
    dp = _bwd_op(gs.float().reshape(1).contiguous(), pred, target, seg, weight, mask)
    return (
        dp if need[0] else None,
        (-dp).to(target.dtype) if need[1] else None,
        None,
        None,
        None,
    )


_fwd_op.register_autograd(_backward, setup_context=_setup)


def token_mse(
    pred: Tensor, target: Tensor, seg: Tensor, weight: Tensor, mask: Tensor | None
) -> Tensor:
    """fp32 sum over tokens of weight[seg] * mask * mean_c((pred - target)^2)."""
    c = pred.shape[-1]
    try:
        return _fwd_op(
            pred.reshape(-1, c).contiguous(),
            target.reshape(-1, c).contiguous(),
            seg,
            weight,
            mask,
        )
    except CompileError:
        return eager_loss.token_mse(pred, target, seg, weight, mask)


register(
    "token_mse",
    "tilelang",
    kernel_or_compile("tilelang", "token_mse", token_mse, shape_key),
    FP16_SUPPORTS,
)
