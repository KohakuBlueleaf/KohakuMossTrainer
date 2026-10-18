"""Global token-count normalisation of the loss sum.

`loss = local_sum * world_factor / global_count`: with gradients averaged over
`world_factor` ranks this equals the global per-token mean.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from kohakumoss_common.calls import identity
from kohakumoss_common.devices import streamed_type


def all_reduce_sum(t: Tensor) -> Tensor:
    """In-place SUM all-reduce over the default process group."""
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t


class TokenNormalizer:
    """Turns a local token count into the global denominator (int64, on device)."""

    def __init__(
        self,
        reduce: Callable[[Tensor], Tensor] = identity,
        world_factor: int = 1,
    ):
        self.reduce = reduce
        self.world_factor = world_factor

    def denominator(self, local_count: int | Tensor, device) -> Tensor:
        """The global count on `device`; a host count reaches an accelerator through
        pinned memory without a host sync."""
        device = torch.device(device)
        if isinstance(local_count, Tensor) and local_count.device.type != "cpu":
            t = local_count.to(device, torch.int64).reshape(1).clone()
        elif streamed_type(device.type):
            host = torch.tensor([int(local_count)], dtype=torch.int64).pin_memory()
            t = host.to(device, non_blocking=True)
        else:
            t = torch.as_tensor(local_count, dtype=torch.int64).reshape(1).clone()
        return self.reduce(t).clamp_min(1)

    def __call__(self, loss_sum: Tensor, denom: Tensor) -> Tensor:
        return loss_sum * self.world_factor / denom.to(loss_sum.dtype)
