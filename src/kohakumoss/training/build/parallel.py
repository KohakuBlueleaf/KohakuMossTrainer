"""Loop hooks for each PARALLEL kind (the training side of kohakumoss.parallel; the
kind's Layout in kohakumoss.training.layouts picks one). DDP and FSDP2 both SUM
gradients over ranks; the loss is a share of the global-token mean."""

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from kohakumoss.diffusion.objective import all_reduce_sum
from kohakumoss.parallel.ddp import (
    AsyncReducer,
    GradReducer,
    broadcast_params,
    wrap_torch_ddp,
)
from kohakumoss.parallel.fsdp import block_units, conditioning_units, shard
from kohakumoss.training.drivers import no_accumulate
from kohakumoss.training.optim.bundle.base import (
    all_reduce_max,
    all_reduce_norm,
    local_grad_norm,
)
from kohakumoss_common.calls import identity


@dataclass
class ParallelHooks:
    """What the loop needs from the parallel layout (single-device defaults)."""

    reduce: Callable = identity
    world_factor: int = 1
    agree: Callable = identity
    reduce_loss: Callable = identity
    grad_norm: Callable = local_grad_norm
    accumulate: Callable = no_accumulate
    before: Callable = lambda: None
    timing: Callable = dict
    ddp: object = None
    reduce_step: Callable | None = None
    forward: object = None


REDUCE_DTYPES = {"fp32": torch.float32, "param": None}
DDP_BACKENDS = ("kohakumoss", "torch")


def reduce_dtype(name: str) -> torch.dtype | None:
    """DDP_REDUCE_DTYPE: 'fp32' sums every gradient in fp32, 'param' in its own dtype."""
    if name not in REDUCE_DTYPES:
        raise ValueError(f"DDP_REDUCE_DTYPE={name!r} not in {tuple(REDUCE_DTYPES)}")
    return REDUCE_DTYPES[name]


def reduce_mean_loss(t: Tensor) -> Tensor:
    """Each rank holds its share of the global-token loss; their sum is the loss."""
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def sum_loss_and_flag(loss: Tensor, bad: Tensor) -> tuple[Tensor, Tensor]:
    """The step's loss (sum of the ranks' shares) and the number of ranks that flagged
    a non-finite gradient, in one SUM all-reduce."""
    both = torch.cat([loss.reshape(1).float(), bad.reshape(1).float()])
    dist.all_reduce(both, op=dist.ReduceOp.SUM)
    return both[:1], both[1:].to(torch.int32)


class _Started:
    def __init__(self, reducer):
        self.reducer = reducer

    def __call__(self, last: bool):
        self.reducer.start(last)
        return contextlib.nullcontext()


def ddp_hooks(built, device: torch.device, reduce: str = "fp32") -> ParallelHooks:
    """SUM reduce of every trainable gradient on the last micro-batch (in fp32, or the
    parameters' dtype: DDP_REDUCE_DTYPE): device gradients in place in GradReducer
    buckets when the plan keeps them until the step, otherwise through the
    AsyncReducer (`ddp`) that the memory wiring hands them to (in-backward steps,
    gradient offload, streamed blocks). Rank 0's trainable device parameters are
    broadcast; streamed masters are one shared host copy already."""
    dtype = reduce_dtype(reduce)
    params = list(built.trainable.trainable_params())
    host = {id(p) for p in built.engines.host_params()}
    local = [p for p in params if id(p) not in host]
    broadcast_params(local, device=device)
    plan = built.plan
    plain = not (plan.fused_step or plan.grad_offload)
    inner = (
        GradReducer(local, average=False, reduce_dtype=dtype)
        if plain and local
        else None
    )
    reducer = AsyncReducer(device, inner=inner, reduce_dtype=dtype)
    return ParallelHooks(
        reduce=all_reduce_sum,
        world_factor=1,
        agree=all_reduce_max,
        reduce_loss=reduce_mean_loss,
        accumulate=_Started(reducer),
        before=reducer.finish,
        timing=reducer.metrics,
        ddp=reducer,
        reduce_step=sum_loss_and_flag,
    )


class _NoSync:
    """torch DDP accumulate context: `no_sync()` on every micro-batch but the last."""

    def __init__(self, ddp):
        self.ddp = ddp

    def __call__(self, last: bool):
        return contextlib.nullcontext() if last else self.ddp.no_sync()


def torch_ddp_hooks(built, device: torch.device, reduce: str = "fp32") -> ParallelHooks:
    """DDP_BACKEND='torch': the backbone under stock torch DDP (`forward`, the module
    the train step calls), gradients summed by its buckets on the last micro-batch."""
    ddp = wrap_torch_ddp(built.backbone, device, reduce_dtype=reduce_dtype(reduce))
    return ParallelHooks(
        reduce=all_reduce_sum,
        world_factor=1,
        agree=all_reduce_max,
        reduce_loss=reduce_mean_loss,
        accumulate=_NoSync(ddp),
        reduce_step=sum_loss_and_flag,
        forward=ddp,
    )


def fsdp_grad_norm(grads: list[Tensor]) -> Tensor:
    """Global L2 norm of sharded (DTensor) grads."""
    norm = torch.nn.utils.get_total_norm(grads, 2.0)
    return norm.full_tensor() if hasattr(norm, "full_tensor") else norm


class _GradientSync:
    """FSDP2 accumulate context: reduce-scatter only on the last micro-batch."""

    def __init__(self, root):
        self.root = root

    def __call__(self, last: bool):
        self.root.set_requires_gradient_sync(last)
        return contextlib.nullcontext()


def fsdp_accumulate(backbone, reduce: str) -> Callable:
    """FSDP_REDUCE: 'step' reduce-scatters once per step (unsharded gradients held over
    the micro-batches), 'micro' after every micro-batch (sharded gradients only)."""
    match reduce:
        case "step":
            return _GradientSync(backbone)
        case "micro":
            return no_accumulate
    raise ValueError(f"FSDP_REDUCE={reduce!r} not in ('step', 'micro')")


def reshard_after_forward(reshard: str) -> bool:
    """FSDP_RESHARD: 'forward' frees each block's gathered parameters after its forward
    and gathers them again in backward (ZeRO-3); 'backward' holds them from the forward
    gather through backward (ZeRO-2: one gather per micro-batch)."""
    match reshard:
        case "forward":
            return True
        case "backward":
            return False
    raise ValueError(f"FSDP_RESHARD={reshard!r} not in ('forward', 'backward')")


def fsdp_hooks(
    backbone, world: int, reduce: str = "step", reshard: str = "forward"
) -> ParallelHooks:
    """Shard each conditioning linear, text-refiner and joint block, then the backbone
    (parameters in their own dtype, FSDP2 SUM of fp32 gradients); gradient sync per
    FSDP_REDUCE (`fsdp_accumulate`), block resharding per FSDP_RESHARD
    (`reshard_after_forward`)."""
    shard(
        backbone,
        block_units(backbone),
        conditioning_units(backbone),
        reshard_after_forward=reshard_after_forward(reshard),
    )
    return ParallelHooks(
        reduce=all_reduce_sum,
        world_factor=1,
        agree=all_reduce_max,
        reduce_loss=reduce_mean_loss,
        grad_norm=fsdp_grad_norm,
        accumulate=fsdp_accumulate(backbone, reduce),
        reduce_step=sum_loss_and_flag,
    )


def pipeline_hooks() -> ParallelHooks:
    """Every rank sees the same data (no count / loss reduction); each holds disjoint
    stage parameters, so the norm and the skip flag are reduced."""
    return ParallelHooks(agree=all_reduce_max, grad_norm=all_reduce_norm)
