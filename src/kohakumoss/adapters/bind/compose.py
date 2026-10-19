"""Shared pieces of bound forwards, chosen at bind time (bias, multiplier, scalar)."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from kohakumoss.adapters.bind import forwards


def has_scalar_param(lyco) -> bool:
    scalar = getattr(lyco, "scalar", None)
    return isinstance(scalar, torch.Tensor) and scalar.requires_grad


def _lin_nobias(x: Tensor, w: Tensor) -> Tensor:
    return F.linear(x, w.to(x.dtype))


def lin_of(t) -> Callable[[Tensor, Tensor], Tensor]:
    """(x, W) -> F.linear in x's dtype, with or without the layer's bias (read per
    call through the target's weak reference)."""
    if t.linear.bias is None:
        return _lin_nobias
    layer = t.layer

    def biased(x: Tensor, w: Tensor) -> Tensor:
        return F.linear(x, w.to(x.dtype), layer().bias.to(x.dtype))

    return biased


def dense_of(linear) -> Callable[[Tensor, Tensor], Tensor]:
    """(x, dW) -> x @ dW^T in x's dtype (no bias)."""
    return _lin_nobias


def _keep(w: Tensor, wp: Tensor) -> Tensor:
    return wp


def mix_of(m: float) -> Callable[[Tensor, Tensor], Tensor]:
    """(W, W') -> W + m (W' - W): the multiplier's interpolation of a whole weight."""
    if m == 1.0:
        return _keep

    def mix(w: Tensor, wp: Tensor) -> Tensor:
        return w + m * (wp - w)

    return mix


def with_scalar(lyco, unit: Callable | None) -> Callable | None:
    """A dW builder times the module's trainable scalar, when it has one."""
    if unit is None or not has_scalar_param(lyco):
        return unit

    def scaled(w: Tensor) -> Tensor:
        return unit(w) * lyco.scalar

    return scaled


def plus_base(t, diff: Callable | None, with_base: Callable | None = None):
    """base(x) + diff(x) (times the trainable scalar), or None without a diff; the
    `with_base` op (the add inside the op) when given and no scalar trains."""
    if diff is None:
        return None
    if has_scalar_param(t.lyco):
        return forwards.plus_base_scalar(t.layer, t.lyco, diff)
    if with_base is not None:
        return forwards.with_base(t.layer, with_base)
    return forwards.plus_base(t.layer, diff)


def merged(t, unit: Callable | None, m: float, merge: Callable | None = None):
    """lin(x, W + m unit(W)), or None without a unit; lin(x, merge(W)) when a W + dW
    op is given (multiplier 1)."""
    if unit is None:
        return None
    lin = lin_of(t)
    if merge is not None:
        return forwards.merged_with_base(t.base_fn, merge, lin)
    if m == 1.0:
        return forwards.merged(t.base_fn, unit, lin)
    return forwards.merged_scaled(t.base_fn, unit, m, lin)
