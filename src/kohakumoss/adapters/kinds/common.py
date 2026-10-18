"""Builders shared by the kind records, and the multiplier-bound (dW, db) builder."""

from collections.abc import Callable

import torch
from torch import Tensor

from kohakumoss.adapters.bind.dora import decomposed

F32 = torch.float32


def names(candidates: tuple[str, ...], tensors: dict) -> tuple[str, ...]:
    """The candidate tensor names present in a layer's tensor set."""
    return tuple(n for n in candidates if n in tensors)


def unit_diff(t) -> Tensor:
    """The module's own get_diff_weight at multiplier 1, fp32, in the layer's shape."""
    return t.lyco.get_diff_weight(1.0)[0].float().reshape(t.lyco.shape)


def diff_set(t, multiplier: float = 1.0) -> dict[str, Tensor]:
    """{"diff": dW (, "diff_b": db)} at `multiplier`."""
    dw, db = t.delta_at(multiplier)()
    out = {"diff": dw}
    if db is not None:
        out["diff_b"] = db
    return out


def _dora_at(t, multiplier: float) -> Callable[[], tuple[Tensor, None]]:
    """DoRA's dW = m (decompose(W + dW) - W)."""

    def build() -> tuple[Tensor, None]:
        w = t.base_fn(F32)
        dw = t.unit_fn().to(w.device)
        delta = decomposed(t, t.backend, w + dw) - w
        return (delta if multiplier == 1.0 else delta * multiplier), None

    return build


def _unit_at(t, multiplier: float) -> Callable[[], tuple[Tensor, None]]:
    if multiplier == 1.0:

        def unit() -> tuple[Tensor, None]:
            return t.unit_fn(), None

        return unit

    def scaled() -> tuple[Tensor, None]:
        return t.unit_fn() * multiplier, None

    return scaled


def delta_builder(t, multiplier: float) -> Callable[[], tuple[Tensor, Tensor | None]]:
    """(dW, db) builder for `t` at a fixed multiplier, fp32 on the adapter device."""
    if t.wd:
        return _dora_at(t, multiplier)
    if t.record.diff_weight_at is not None:
        special = t.record.diff_weight_at(t, multiplier)
        if special is not None:
            return special
    return _unit_at(t, multiplier)
