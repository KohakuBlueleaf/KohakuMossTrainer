"""Recorded `Bound` calls for the prewarm replay (`recording()`): every tensor argument
as a `Snap` (layout, and its values up to `KEEP_NUMEL` elements). See
docs/internals/tuning.md#prewarm.
"""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor

KEEP_NUMEL = 1 << 16
INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True, eq=False)
class Snap:
    """A tensor argument as recorded: layout, and its values when small."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool
    values: Tensor | None = None
    span: tuple[int, int] | None = None

    @property
    def floating(self) -> bool:
        return self.dtype.is_floating_point or self.dtype.is_complex

    def tensor(self, rows: int | None = None) -> Tensor:
        """A fresh detached tensor of this layout on the recorded device, `rows` long in
        dim 0 when given (kept values resampled in order, rows repeated)."""
        if self.values is not None:
            if rows is None or rows == self.shape[0]:
                return self.values.clone()
            idx = torch.arange(rows, device=self.device) * self.shape[0] // rows
            return self.values[idx]
        shape = self.shape if rows is None else (rows, *self.shape[1:])
        if self.span is not None:
            lo, hi = self.span
            out = torch.randint(lo, min(hi + 1, INT64_MAX), shape, device=self.device)
            return out.to(self.dtype)
        return torch.randn(shape, device=self.device).to(self.dtype)


def snapshot(a):
    """`a` with every tensor (also inside tuples / lists) replaced by its `Snap`."""
    match a:
        case Tensor():
            return _snap(a)
        case tuple() | list():
            return type(a)(snapshot(x) for x in a)
    return a


def _snap(t: Tensor) -> Snap:
    layout = (tuple(t.shape), t.dtype, t.device, t.requires_grad)
    if t.numel() <= KEEP_NUMEL:
        return Snap(*layout, values=t.detach().clone())
    if t.dtype.is_floating_point or t.dtype.is_complex:
        return Snap(*layout)
    return Snap(*layout, span=(int(t.min()), int(t.max())))


@dataclass
class Call:
    """One recorded call of a `Bound` op: the op, its arguments (tensors as `Snap`s)
    and the autocast state (device type, dtype) it ran under."""

    op: object
    args: tuple
    kwargs: dict
    autocast: tuple[str, torch.dtype] | None = None


RECORDS: list[list[Call]] = []
_CHANGED: list[Callable[[], None]] = []


def on_change(fn: Callable[[], None]) -> None:
    """Register `fn`, run whenever a recording starts or ends (callers rebind)."""
    _CHANGED.append(fn)


def _changed() -> None:
    for fn in _CHANGED:
        fn()


@contextmanager
def recording():
    """Record every `Bound` call made inside (each answered by the eager reference,
    nothing built or tuned); yields the list of `Call`s."""
    calls: list[Call] = []
    RECORDS.append(calls)
    _changed()
    try:
        yield calls
    finally:
        RECORDS.pop()
        _changed()


def record(op, args: tuple, kwargs: dict) -> None:
    """Append a snapshot of this call of `op` to the innermost recording."""
    kind = next((a.device.type for a in args if isinstance(a, Tensor)), "cpu")
    cast = (
        (kind, torch.get_autocast_dtype(kind))
        if torch.is_autocast_enabled(kind)
        else None
    )
    snaps = {k: snapshot(v) for k, v in kwargs.items()}
    RECORDS[-1].append(Call(op, snapshot(args), snaps, cast))
