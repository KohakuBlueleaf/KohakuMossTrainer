"""FSDP2 sharding: `fully_shard` per joint block (adapter modules inside their block),
then the root, with a bf16/fp16 param dtype and fp32 gradient reduction."""

import contextlib

import torch
from torch import nn

try:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        fully_shard,
    )
    from torch.distributed.tensor import DTensor, distribute_tensor
except ImportError:
    fully_shard = None
    MixedPrecisionPolicy = None
    CPUOffloadPolicy = None
    DTensor = ()


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
    `sum_grads`: reduce gradients by SUM over ranks (else FSDP's mean)."""
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


def _units(model: nn.Module) -> list[nn.Module]:
    return [
        m for m in model.modules() if hasattr(m, "unshard") and hasattr(m, "reshard")
    ]


@contextlib.contextmanager
def unsharded(model: nn.Module):
    """Every FSDP unit of `model` holds full parameters inside (collective: every rank
    enters); outside, the shards are the parameters again."""
    units = _units(model)
    for unit in units:
        unit.unshard()
    try:
        yield
    finally:
        for unit in units:
            unit.reshard()


def full_tensors(state: dict) -> dict:
    """DTensor values gathered to full tensors (collective, same key order on every
    rank); other values unchanged."""
    return {
        k: v.full_tensor() if isinstance(v, DTensor) else v for k, v in state.items()
    }


def shard_like(state: dict, templates: dict) -> dict:
    """Full tensors of `state` re-sharded like the DTensors of `templates`."""
    out = dict(state)
    for k, t in templates.items():
        if isinstance(t, DTensor) and k in state:
            full = state[k].to(t.device, t.dtype)
            out[k] = distribute_tensor(full, t.device_mesh, t.placements)
    return out


def optimizer_state(model: nn.Module, optimizer) -> dict:
    """The optimiser's full (unsharded, host) state dict keyed by parameter name."""
    return get_optimizer_state_dict(
        model,
        optimizer,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


def load_optimizer_state(model: nn.Module, optimizer, state: dict) -> None:
    """Re-shard a full optimiser state dict into `optimizer` (every rank)."""
    set_optimizer_state_dict(
        model,
        optimizer,
        optim_state_dict=state,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=False),
    )
