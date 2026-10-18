"""Contexts and hooks around adapter state changes: deterministic evaluation for
exports, and re-binding the owning backbone."""

from contextlib import contextmanager

from torch import nn

from kohakumoss.adapters.train import tlora


@contextmanager
def evaluated(targets):
    """Every module in eval mode (full rank, every DyLoRA block) with the T-LoRA masks
    cleared; both restored on exit. Bound forwards are not re-bound."""
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


def rebind_owner(model: nn.Module) -> None:
    """Re-bind the ops of every bound backbone in `model`, so fused linear+epilogue
    layers are re-selected after adapters are attached, merged or removed."""
    for m in model.modules():
        if hasattr(m, "bind_ops") and getattr(m, "ops", None) is not None:
            m.bind_ops()
