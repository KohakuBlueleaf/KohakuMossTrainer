"""Contexts and hooks around adapter state changes: deterministic evaluation for
exports, scratch factors for tier prewarm, and re-binding the owning backbone."""

from contextlib import contextmanager
from functools import partial

import torch
from torch import nn

from kohakumoss.adapters import tlora
from kohakumoss.adapters.bind.race import TierSlot


@contextmanager
def evaluated(targets):
    """Every module in eval mode (no rank dropout, every DyLoRA block) and no T-LoRA
    mask installed; both restored on exit. Bound forwards are not re-bound."""
    masks = tlora.saved_masks()
    tlora.clear_masks()
    modes = [(t.lyco, t.lyco.training) for t in targets]
    for lyco, _ in modes:
        nn.Module.train(lyco, False)
    try:
        yield
    finally:
        for lyco, mode in modes:
            nn.Module.train(lyco, mode)
        tlora.restore_masks(masks)


@contextmanager
def perturbed(lyco: nn.Module, std: float = 0.02):
    """The module's parameters replaced by random scratch values (restored on exit,
    gradients dropped), so a tier race sees non-zero references."""
    params = list(lyco.parameters())
    saved = [p.data for p in params]
    gen = torch.Generator(device="cpu").manual_seed(0)
    for p in params:
        noise = torch.randn(p.shape, generator=gen, dtype=torch.float32) * std
        p.data = noise.to(device=p.device, dtype=p.dtype) + p.data
    try:
        yield
    finally:
        for p, data in zip(params, saved):
            p.data = data
            p.grad = None


def slots_of(fn) -> list[TierSlot]:
    """Every TierSlot inside a bound forward (nested partials)."""
    if isinstance(fn, TierSlot):
        return [fn]
    if not isinstance(fn, partial):
        return []
    found = slots_of(fn.func)
    for a in fn.args:
        found += slots_of(a)
    return found


HALF = (torch.float16, torch.bfloat16)


def compute_dtype_of(modules) -> torch.dtype | None:
    """The one 16-bit matmul dtype of the linears in `modules` (their weight or
    MossLinear input dtype), or None for an all-fp32 model."""
    seen = set()
    for m in modules:
        if isinstance(m, nn.Linear):
            weight = getattr(m, "weight", None)
            seen.add(getattr(m, "input_dtype", None))
            seen.add(weight.dtype if isinstance(weight, torch.Tensor) else None)
    half = seen & set(HALF)
    if len(half) > 1:
        raise ValueError(f"linears mix fp16 and bf16: {sorted(map(str, half))}")
    return next(iter(half), None)


def rebind_owner(model: nn.Module) -> None:
    """Re-bind the ops of every bound backbone in `model`, so fused linear+epilogue
    layers are re-selected after adapters are attached, merged or removed."""
    for m in model.modules():
        if hasattr(m, "bind_ops") and getattr(m, "ops", None) is not None:
            m.bind_ops()
