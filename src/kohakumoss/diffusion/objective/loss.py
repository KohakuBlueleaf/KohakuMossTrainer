"""Token losses: fp32 sum over image tokens of weight[seg] * mask * per-token error.

`masked` is fixed at construction and selects the reduction; the mask argument is
None for unmasked losses.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


def _plain_sum(err: Tensor, seg: Tensor, weight: Tensor, mask) -> Tensor:
    return (err * weight.float().index_select(0, seg)).sum(dtype=torch.float32)


def _masked_sum(err: Tensor, seg: Tensor, weight: Tensor, mask: Tensor) -> Tensor:
    w = weight.float().index_select(0, seg) * mask.float()
    return (err * w).sum(dtype=torch.float32)


def _reducer(masked: bool) -> Callable:
    return _masked_sum if masked else _plain_sum


class MSELoss:
    """mean_c((pred - target)^2) per token, through the resolved `token_mse` op."""

    def __init__(self, token_mse: Callable, masked: bool = False):
        self.token_mse = token_mse

    def __call__(self, pred, target, seg, weight, mask) -> Tensor:
        return self.token_mse(pred, target, seg, weight, mask)


class HuberLoss:
    """mean_c(huber(pred - target, delta)) per token (exact Huber)."""

    def __init__(self, token_mse: Callable, delta: float = 1.0, masked: bool = False):
        self.delta = delta
        self.reduce = _reducer(masked)

    def __call__(self, pred, target, seg, weight, mask) -> Tensor:
        err = F.huber_loss(
            pred.float(), target.float(), reduction="none", delta=self.delta
        ).mean(-1)
        return self.reduce(err, seg, weight, mask)


class L1Loss:
    """mean_c(|pred - target|) per token."""

    def __init__(self, token_mse: Callable, masked: bool = False):
        self.reduce = _reducer(masked)

    def __call__(self, pred, target, seg, weight, mask) -> Tensor:
        err = (pred.float() - target.float()).abs().mean(-1)
        return self.reduce(err, seg, weight, mask)
