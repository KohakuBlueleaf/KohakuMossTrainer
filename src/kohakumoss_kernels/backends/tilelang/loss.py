"""TileLang token MSE: sum over tokens of weight[seg] * mask * mean_c((pred - target)^2).

Forward writes one fp32 partial per program row lane, summed in torch; `weight` and
`mask` are data and get no gradient.
"""

import tilelang.language as T
import torch
from torch import Tensor
from torch.library import custom_op

from kohakumoss_kernels.backends.compile.race import raced, shape_key
from kohakumoss_kernels.backends.eager import loss as eager_loss
from kohakumoss_kernels.backends.tilelang._blockops import DTYPE, CompileError, compiled
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.registry import register

__all__ = ["token_mse"]

F32 = "float32"
BM, THREADS = 16, 128


def _fwd_prim(c, dtype):
    n = T.dynamic("n")
    b = T.dynamic("b")
    pp = T.dynamic("p")

    @T.prim_func
    def main(
        p: T.Tensor((n, c), dtype),
        t: T.Tensor((n, c), dtype),
        seg: T.Tensor((n,), "int32"),
        w: T.Tensor((b,), F32),
        m: T.Tensor((n,), F32),
        part: T.Tensor((pp, BM), F32),
    ):
        with T.Kernel(T.ceildiv(n, BM), threads=THREADS) as bx:
            sq = T.alloc_fragment((BM, c), F32)
            err = T.alloc_fragment((BM,), F32)
            for i, j in T.Parallel(BM, c):
                rr = T.min(bx * BM + i, n - 1)
                d = T.cast(p[rr, j], F32) - T.cast(t[rr, j], F32)
                sq[i, j] = d * d
            T.reduce_sum(sq, err, dim=1)
            for i in T.Parallel(BM):
                row = bx * BM + i
                rr = T.min(row, n - 1)
                part[bx, i] = T.if_then_else(
                    row < n, err[i] / c * w[seg[rr]] * m[rr], 0.0
                )

    return main


def _bwd_prim(c, dtype):
    n = T.dynamic("n")
    b = T.dynamic("b")

    @T.prim_func
    def main(
        gs: T.Tensor((1,), F32),
        p: T.Tensor((n, c), dtype),
        t: T.Tensor((n, c), dtype),
        seg: T.Tensor((n,), "int32"),
        w: T.Tensor((b,), F32),
        m: T.Tensor((n,), F32),
        dp: T.Tensor((n, c), dtype),
    ):
        with T.Kernel(T.ceildiv(n, BM), threads=THREADS) as bx:
            for i, j in T.Parallel(BM, c):
                row = bx * BM + i
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


@custom_op("kohakumoss::f6tl_token_mse_fwd", mutates_args=())
def _fwd_op(
    pred: Tensor, target: Tensor, seg: Tensor, weight: Tensor, mask: Tensor | None
) -> Tensor:
    n, c = pred.shape
    part = torch.empty(-(-n // BM), BM, device=pred.device, dtype=torch.float32)
    kernel = compiled(_fwd_prim, pred.device, c, DTYPE[pred.dtype])
    kernel(
        pred,
        target,
        seg.to(torch.int32).contiguous(),
        weight.float().contiguous(),
        _mask(mask, n, pred.device),
        part,
    )
    return part.sum()


@_fwd_op.register_fake
def _(pred, target, seg, weight, mask):
    return pred.new_empty((), dtype=torch.float32)


@custom_op("kohakumoss::f6tl_token_mse_bwd", mutates_args=())
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
    kernel = compiled(_bwd_prim, pred.device, c, DTYPE[pred.dtype])
    kernel(
        gs,
        pred,
        target,
        seg.to(torch.int32).contiguous(),
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
    raced("tilelang", "token_mse", token_mse, shape_key),
    requires(kinds=("cuda",), min_sm=70, dtypes=(torch.float16,)),
)
