"""Shared pieces of bound forwards, chosen at bind time (bias, multiplier, scalar,
the layer's matmul precision)."""

from functools import partial

import torch

from kohakumoss.adapters.bind import fused


def has_scalar_param(lyco) -> bool:
    scalar = getattr(lyco, "scalar", None)
    return isinstance(scalar, torch.Tensor) and scalar.requires_grad


def _input_dtype(linear):
    return getattr(linear, "input_dtype", None)


def lin_of(t):
    """F.linear with or without the layer's bias, in the layer's matmul precision
    (16-bit operands into fp32 for a MossLinear with an input dtype)."""
    dtype = _input_dtype(t.linear)
    match (dtype is not None, t.linear.bias is not None):
        case (True, True):
            return partial(fused.lin_wide_bias, t.linear, dtype)
        case (True, False):
            return partial(fused.lin_wide, dtype)
        case (False, True):
            return partial(fused.lin_bias, t.linear)
    return fused.lin_nobias


def dense_of(linear):
    """(x, dW) -> x @ dW^T in the layer's matmul precision (no bias)."""
    dtype = _input_dtype(linear)
    return fused.lin_nobias if dtype is None else partial(fused.lin_wide, dtype)


def scale_of(m: float):
    return fused.same if m == 1.0 else partial(fused.times, m)


def mix_of(m: float):
    return fused.keep if m == 1.0 else partial(fused.interpolate, m)


def scalared(lyco, unit):
    """A dW builder times the module's trainable scalar, when it has one."""
    if unit is None or not has_scalar_param(lyco):
        return unit
    return partial(fused.with_scalar, lyco, unit)


def added(t, diff):
    """base(x) + diff(x) (times the trainable scalar), or None without a diff."""
    if diff is None:
        return None
    if has_scalar_param(t.lyco):
        return partial(fused.add_base_scalar, t.linear, t.lyco, diff)
    return partial(fused.add_base, t.linear, diff)


def merged(t, unit, m: float):
    """lin(x, W + m * unit(W)), or None without a unit."""
    if unit is None:
        return None
    return partial(fused.merged_forward, t.base_fn, unit, scale_of(m), lin_of(t))
