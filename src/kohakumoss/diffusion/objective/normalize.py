"""Global token-count normalisation of the loss sum.

`loss = local_sum * world_factor / global_count`: with gradients averaged over
`world_factor` ranks this equals the global per-token mean.
"""

from collections.abc import Callable

import torch
from torch import Tensor


def identity_reduce(t: Tensor) -> Tensor:
    return t


def all_reduce_sum(t: Tensor) -> Tensor:
    """In-place SUM all-reduce over the default process group."""
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t


class TokenNormalizer:
    """Turns a local token count into the global denominator (int64, on device)."""

    def __init__(
        self,
        reduce: Callable[[Tensor], Tensor] = identity_reduce,
        world_factor: int = 1,
    ):
        self.reduce = reduce
        self.world_factor = world_factor

    def denominator(self, local_count: int | Tensor, device) -> Tensor:
        t = torch.as_tensor(local_count, dtype=torch.int64, device=device).reshape(1)
        return self.reduce(t.clone()).clamp_min(1)

    def __call__(self, loss_sum: Tensor, denom: Tensor) -> Tensor:
        return loss_sum * self.world_factor / denom.to(loss_sum.dtype)
