"""FSDP2 sharding: `fully_shard` per joint block (adapter modules inside their block),
then the root, with a bf16/fp16 param dtype and fp32 gradient reduction."""

import torch
from torch import nn

try:
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        fully_shard,
    )
except ImportError:
    fully_shard = None
    MixedPrecisionPolicy = None
    CPUOffloadPolicy = None


def available() -> bool:
    return fully_shard is not None


def shard(
    model: nn.Module,
    blocks: list[nn.Module] | nn.ModuleList,
    param_dtype: torch.dtype | None = None,
    reduce_dtype: torch.dtype = torch.float32,
    reshard_after_forward: bool = True,
    cpu_offload: bool = False,
    mesh=None,
    sum_grads: bool = True,
) -> nn.Module:
    """Shard `blocks` one unit each, then `model`; returns `model` (sharded in place).

    `sum_grads` reduces by SUM (the loss is already normalised by the global token
    count) instead of FSDP's default mean over ranks.
    """
    if fully_shard is None:
        raise RuntimeError("torch.distributed.fsdp.fully_shard is not available")
    kw = {}
    if param_dtype is not None:
        kw["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
    if cpu_offload:
        kw["offload_policy"] = CPUOffloadPolicy(pin_memory=True)
    if mesh is not None:
        kw["mesh"] = mesh
    units = list(blocks) + [model]
    for unit in units:
        fully_shard(unit, reshard_after_forward=reshard_after_forward, **kw)
    if sum_grads:
        for unit in units:
            unit.set_gradient_divide_factor(1.0)
    return model
