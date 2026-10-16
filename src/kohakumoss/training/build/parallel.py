"""Loop hooks for each PARALLEL kind (the training side of kohakumoss.parallel).

DDP reduces adapter / full grads with a SUM all-reduce, so the global-token loss needs
no world factor; FSDP2 averages, so the loss carries `world`.
"""

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from kohakumoss.diffusion.objective import all_reduce_sum, identity_reduce
from kohakumoss.registry import import_object
from kohakumoss.training.drivers import no_accumulate
from kohakumoss.training.optim.bundle import (
    all_reduce_max,
    all_reduce_norm,
    local_grad_norm,
    no_agree,
)

GRAD_REDUCER = "kohakumoss.parallel.ddp.GradReducer"
BROADCAST = "kohakumoss.parallel.ddp.broadcast_params"
FSDP_SHARD = "kohakumoss.parallel.fsdp.shard"


@dataclass
class ParallelHooks:
    """What the loop needs from the parallel layout (single-device defaults)."""

    reduce: Callable = identity_reduce
    world_factor: int = 1
    agree: Callable = no_agree
    reduce_loss: Callable = no_agree
    grad_norm: Callable = local_grad_norm
    accumulate: Callable = no_accumulate
    before: Callable = lambda: None


def reduce_mean_loss(t: Tensor) -> Tensor:
    """Each rank holds its share of the global-token loss; their sum is the loss."""
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


class _Armed:
    def __init__(self, reducer):
        self.reducer = reducer

    def __call__(self, last: bool):
        self.reducer.arm(last)
        return contextlib.nullcontext()


def ddp_hooks(params: list) -> ParallelHooks:
    """Bucketed SUM reduce of `params`' grads on the last micro-batch."""
    import_object(BROADCAST)(params)
    reducer = import_object(GRAD_REDUCER)(params, average=False)
    return ParallelHooks(
        reduce=all_reduce_sum,
        world_factor=1,
        agree=all_reduce_max,
        reduce_loss=reduce_mean_loss,
        accumulate=_Armed(reducer),
        before=reducer.finish,
    )


def fsdp_grad_norm(grads: list[Tensor]) -> Tensor:
    """Global L2 norm of sharded (DTensor) grads."""
    norm = torch.nn.utils.get_total_norm(grads, 2.0)
    return norm.full_tensor() if hasattr(norm, "full_tensor") else norm


def fsdp_hooks(backbone, compute: torch.dtype, world: int) -> ParallelHooks:
    """Shard each joint block, then the backbone (FSDP2, fp32 gradient reduction)."""
    import_object(FSDP_SHARD)(backbone, backbone.layers, param_dtype=compute)
    return ParallelHooks(
        reduce=all_reduce_sum,
        world_factor=world,
        agree=all_reduce_max,
        reduce_loss=lambda t: reduce_mean_loss(t).div_(world),
        grad_norm=fsdp_grad_norm,
    )


def pipeline_hooks() -> ParallelHooks:
    """Every rank sees the same data (no count / loss reduction); each holds disjoint
    stage parameters, so the norm and the skip flag are reduced."""
    return ParallelHooks(agree=all_reduce_max, grad_norm=all_reduce_norm)


def build_hooks(kind: str, built, compute: torch.dtype, world: int) -> ParallelHooks:
    match kind:
        case "none" | "threaded":
            return ParallelHooks()
        case "ddp":
            return ddp_hooks(list(built.trainable.trainable_params()))
        case "fsdp":
            return fsdp_hooks(built.backbone, compute, world)
        case "pipeline":
            return pipeline_hooks()
        case _:
            raise SystemExit(f"PARALLEL={kind!r} unknown")
