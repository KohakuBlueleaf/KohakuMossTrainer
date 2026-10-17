"""Builders shared by the kind records, and the multiplier-bound (dW, db) builder."""

from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor

from kohakumoss.adapters.bind.dora import decomposed

F32 = torch.float32


def bound(fn: Callable, t) -> Callable[[], Tensor]:
    """`fn(t)` as a zero-argument builder bound to its Target."""
    return partial(fn, t)


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


def _unit_only(t) -> tuple[Tensor, None]:
    return t.unit_fn(), None


def _unit_times(t, multiplier: float) -> tuple[Tensor, None]:
    return t.unit_fn() * multiplier, None


def _dora(t, scale: Callable) -> tuple[Tensor, None]:
    w = t.base_fn(F32)
    dw = t.unit_fn().to(w.device)
    return scale(decomposed(t, t.backend, w + dw) - w), None


def _times(multiplier: float, x: Tensor) -> Tensor:
    return x * multiplier


def _same(x: Tensor) -> Tensor:
    return x


def delta_builder(t, multiplier: float) -> Callable[[], tuple[Tensor, Tensor | None]]:
    """(dW, db) builder for `t` at a fixed multiplier, fp32 on the adapter device."""
    one = multiplier == 1.0
    if t.wd:
        return partial(_dora, t, _same if one else partial(_times, multiplier))
    if t.record.at is not None:
        special = t.record.at(t, multiplier)
        if special is not None:
            return special
    return partial(_unit_only, t) if one else partial(_unit_times, t, multiplier)
