"""FSDP2 sharding: `fully_shard` per conditioning linear and per block unit (joint
blocks and text-refiner blocks, adapter modules inside their module), then the root;
every parameter gathered in its own dtype, fp32 gradient reduction and explicit
next-unit prefetch. See docs/guides/multi-gpu.md."""

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


def block_units(backbone: nn.Module) -> list[nn.Module]:
    """The backbone's per-unit modules in forward order: the text-refiner blocks, then
    the joint blocks."""
    text = getattr(getattr(backbone, "text_embedder", None), "blocks", None)
    return [*(text or ()), *backbone.layers]


def conditioning_units(backbone: nn.Module) -> list[nn.Module]:
    """The backbone's per-sample conditioning linears (the KEEP_FP32 modules), each
    called through its own `__call__`; none for a model without them."""
    linears = getattr(backbone, "conditioning_linears", None)
    return list(linears()) if linears is not None else []


def shard(
    model: nn.Module,
    blocks: list[nn.Module] | nn.ModuleList,
    own: tuple[nn.Module, ...] | list[nn.Module] = (),
    reduce_dtype: torch.dtype = torch.float32,
    reshard_after_forward: bool = True,
    cpu_offload: bool = False,
    mesh=None,
    sum_grads: bool = True,
    prefetch: int = 1,
) -> nn.Module:
    """Shard each module of `own` (root-level modules called through `__call__`, e.g.
    `conditioning_units`) and of `blocks` (in forward order) as one unit, then `model`;
    returns `model` (sharded in place). Every parameter is gathered in its own dtype;
    gradients are reduced in `reduce_dtype`. `sum_grads`: gradients are a plain SUM
    over ranks (else FSDP's mean). `prefetch`: block units whose all-gather is issued
    ahead, forward and backward; the root prefetches `own` and the first block. Forward
    inputs keep their dtype (sigma stays fp32); the root and the `own` units keep their
    parameters gathered between forward and backward. See docs/guides/multi-gpu.md."""
    if fully_shard is None:
        raise RuntimeError("torch.distributed.fsdp.fully_shard is not available")
    kw = {
        "mp_policy": MixedPrecisionPolicy(
            reduce_dtype=reduce_dtype, cast_forward_inputs=False
        )
    }
    if cpu_offload:
        kw["offload_policy"] = CPUOffloadPolicy(pin_memory=True)
    if mesh is not None:
        kw["mesh"] = mesh
    own, units = list(own), list(blocks)
    for unit in own:
        fully_shard(unit, reshard_after_forward=False, **kw)
    for unit in units:
        fully_shard(unit, reshard_after_forward=reshard_after_forward, **kw)
    fully_shard(model, reshard_after_forward=False, **kw)
    if sum_grads:
        for unit in [*own, *units, model]:
            unit.set_gradient_divide_factor(1.0)
            unit.set_force_sum_reduction_for_comms(True)
    set_prefetch(units, prefetch)
    ahead = [*own, *units[:1]] if prefetch > 0 else []
    if ahead:
        model.set_modules_to_forward_prefetch(ahead)
    return model


def set_prefetch(units: list[nn.Module], depth: int) -> None:
    """Unit i all-gathers units i+1..i+depth in forward and i-1..i-depth in backward."""
    if depth <= 0:
        return
    n = len(units)
    for i, unit in enumerate(units):
        ahead = units[i + 1 : min(n, i + 1 + depth)]
        behind = units[max(0, i - depth) : i][::-1]
        if ahead:
            unit.set_modules_to_forward_prefetch(ahead)
        if behind:
            unit.set_modules_to_backward_prefetch(behind)


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
    """DTensor values gathered to full host tensors one at a time (collective, same key
    order on every rank; the device holds one gathered tensor at a time); other values
    unchanged."""
    out = {}
    for k, v in state.items():
        if isinstance(v, DTensor):
            full = v.full_tensor()
            out[k] = full.detach().to("cpu", copy=True)
            del full
        else:
            out[k] = v
    return out


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
