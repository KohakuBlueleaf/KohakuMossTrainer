"""Token-level MSE with per-sample weights and an optional per-token mask."""

from torch import Tensor

from kohakumoss_kernels.backends.eager._acc import acc, acc_dtype

__all__ = ["token_mse"]


def token_mse(
    pred: Tensor, target: Tensor, seg: Tensor, weight: Tensor, mask: Tensor | None
) -> Tensor:
    """Sum over tokens of weight[seg] * mask * mean_c((pred - target)^2), >= fp32."""
    dt = acc_dtype(pred, target)
    err = (pred.to(dt) - target.to(dt)).square().mean(-1)
    w = acc(weight, dt).index_select(0, seg)
    if mask is not None:
        w = w * acc(mask, dt)
    return (err * w).sum()
