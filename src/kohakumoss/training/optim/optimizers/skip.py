"""Skipped updates of the fused per-parameter step: a parameter whose found_inf flag is
set keeps its value and its optimiser state. Working copies and masked stores for
KohakuMoss's own optimisers, device masks and host-state restores for any other. See
docs/internals/offload.md (fused step, non-finite gradients)."""

import torch
from torch import Tensor
from torch.optim import Optimizer


def working(t: Tensor, dtype: torch.dtype) -> Tensor:
    """`t` in `dtype` (`t` itself when it already is)."""
    return t.to(dtype)


def working_copy(t: Tensor, dtype: torch.dtype) -> Tensor:
    """A new copy of `t` in `dtype`."""
    return t.to(dtype, copy=True)


def store(dst: Tensor, new: Tensor) -> None:
    dst.copy_(new)


def store_unless(skip: Tensor, dst: Tensor, new: Tensor) -> None:
    """dst = new, except where the 0-d bool `skip` is set."""
    dst.copy_(torch.where(skip, dst, new.to(dst.dtype)))


def put_back(skip: Tensor, dst: Tensor, old: Tensor) -> None:
    """dst = old where the 0-d bool `skip` is set."""
    dst.copy_(torch.where(skip, old, dst))


def write_unless(write, skip: Tensor, p: Tensor, new: Tensor, comp) -> None:
    """A rounding writer's `write(p, new, comp)`, kept only where `skip` is unset."""
    out = torch.empty_like(p)
    write(out, new, comp)
    store_unless(skip, p, out)


def write_unless_comp(write, skip: Tensor, p: Tensor, new: Tensor, comp: Tensor):
    """`write_unless` for a writer that also updates its compensation `comp`."""
    out, kept = torch.empty_like(p), comp.clone()
    write(out, new, kept)
    store_unless(skip, p, out)
    store_unless(skip, comp, kept)


def on_device(value, device: torch.device) -> bool:
    return isinstance(value, Tensor) and value.device == device


def host_state(opt: Optimizer, p: Tensor) -> tuple[dict | None, dict]:
    """What a device mask cannot put back after `opt` steps `p`: the state entries that
    are plain values or tensors off `p`'s device (step counts; None when `p` has no
    state yet) and the group's values."""
    state = opt.state[p]
    kept = (
        {
            k: v.clone() if isinstance(v, Tensor) else v
            for k, v in state.items()
            if not on_device(v, p.device)
        }
        if state
        else None
    )
    group = {k: v for k, v in opt.param_groups[0].items() if k != "params"}
    return kept, group


def restore_host(opt: Optimizer, p: Tensor, saved: tuple[dict | None, dict]) -> None:
    """Put back `host_state(opt, p)`; a state the step created is dropped."""
    kept, group = saved
    opt.param_groups[0].update(group)
    if kept is None:
        opt.state.pop(p, None)
    else:
        opt.state[p].update(kept)
