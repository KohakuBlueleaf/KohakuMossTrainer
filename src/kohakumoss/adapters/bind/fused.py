"""Bound forwards over the LyCORIS fused kernels with a pinned backend, and the
forward shells the raced ops of `bind.ops` plug into. State first, runtime tensor
last; one function per variant (see docs/internals/adapter-binding.md).
"""

from functools import partial

import torch
import torch.nn.functional as F
from lycoris.kernels.autograd import (
    apply_dora,
    boft_bypass_diff,
    boft_diff_weight,
    diag_oft_bypass_diff,
    diag_oft_diff_weight,
    glora_diff_weight,
    ia3_bypass,
    ia3_diff_weight,
    locon_bypass_diff,
    locon_diff_weight,
    loha_bypass_diff,
    loha_diff_weight,
    lokr_kron_bypass,
    lokr_kron_weight,
)
from torch import Tensor

from kohakumoss.model.components.wide import wide_linear


def _attr(name, lyco, dtype):
    return getattr(lyco, name).to(dtype)


def _attr_raw(name, lyco, dtype):
    return getattr(lyco, name)


def _none(lyco, dtype):
    return None


def kron_getters(lyco, cast: bool = True) -> tuple:
    """Six getters (lyco, dtype) -> w1, w1a, w1b, w2, w2a, w2b, chosen once per layer."""
    get = _attr if cast else _attr_raw
    side1 = (
        (partial(get, "lokr_w1"), _none, _none)
        if lyco.use_w1
        else (_none, partial(get, "lokr_w1_a"), partial(get, "lokr_w1_b"))
    )
    side2 = (
        (partial(get, "lokr_w2"), _none, _none)
        if lyco.use_w2
        else (_none, partial(get, "lokr_w2_a"), partial(get, "lokr_w2_b"))
    )
    return side1 + side2


def kron_factors(lyco, getters, dtype=None) -> tuple:
    return tuple(g(lyco, dtype) for g in getters)


def lin_bias(linear, x: Tensor, w: Tensor) -> Tensor:
    return F.linear(x, w.to(x.dtype), linear.bias.to(x.dtype))


def lin_nobias(x: Tensor, w: Tensor) -> Tensor:
    return F.linear(x, w.to(x.dtype))


def lin_wide(dtype, x: Tensor, w: Tensor) -> Tensor:
    """x @ w^T on `dtype` (16-bit) operands, fp32 accumulation and output."""
    return wide_linear(x, w, dtype)


def lin_wide_bias(linear, dtype, x: Tensor, w: Tensor) -> Tensor:
    return wide_linear(x, w, dtype) + linear.bias.float()


def same(t: Tensor) -> Tensor:
    return t


def times(multiplier: float, t: Tensor) -> Tensor:
    return t * multiplier


def keep(w: Tensor, wp: Tensor) -> Tensor:
    return wp


def interpolate(multiplier: float, w: Tensor, wp: Tensor) -> Tensor:
    return w + multiplier * (wp - w)


def add_base(linear, diff, x):
    return linear._impl(x) + diff(x)


def add_base_scalar(linear, lyco, diff, x):
    return linear._impl(x) + diff(x) * lyco.scalar.to(x.dtype)


def lora_diff(lyco, gamma, backend, x):
    d = lyco.lora_down.weight.to(x.dtype)
    u = lyco.lora_up.weight.to(x.dtype)
    return locon_bypass_diff(x, d, u, gamma, backend)


def loha_diff(lyco, gamma, backend, x):
    dt = x.dtype
    return loha_bypass_diff(
        x,
        lyco.hada_w1_b.to(dt),
        lyco.hada_w1_a.to(dt),
        lyco.hada_w2_b.to(dt),
        lyco.hada_w2_a.to(dt),
        gamma,
        backend,
    )


def lokr_diff(lyco, getters, scale, backend, x):
    return lokr_kron_bypass(
        x, *kron_factors(lyco, getters, x.dtype), None, scale, backend
    )


def oft_bypass(linear, lyco, backend, x):
    y = linear._impl(x)
    delta = diag_oft_bypass_diff(
        y, lyco.oft_blocks, None, lyco.constraint, False, backend
    )
    return y + delta.to(y.dtype)


def oft_bypass_rescaled(linear, lyco, backend, x):
    y = linear._impl(x)
    delta = diag_oft_bypass_diff(
        y, lyco.oft_blocks, lyco.rescale, lyco.constraint, False, backend
    )
    return y + delta.to(y.dtype)


def boft_bypass(linear, lyco, multiplier, backend, x):
    y = linear._impl(x)
    delta = boft_bypass_diff(
        y, lyco.oft_blocks, None, lyco.constraint, multiplier, False, backend
    )
    return y + delta.to(y.dtype)


def boft_bypass_rescaled(linear, lyco, multiplier, backend, x):
    y = linear._impl(x)
    delta = boft_bypass_diff(
        y, lyco.oft_blocks, lyco.rescale, lyco.constraint, multiplier, False, backend
    )
    return y + delta.to(y.dtype)


def post_add(linear, delta, x):
    """y + dy(y) for a rotation whose dy is a function of the base output."""
    y = linear._impl(x)
    return y + delta(y).to(y.dtype)


def ia3_bypass_out(linear, lyco, multiplier, backend, x):
    y = linear._impl(x)
    return ia3_bypass(y, lyco.weight.to(y.dtype), -1, multiplier, False, backend)


def ia3_bypass_in(linear, lyco, multiplier, backend, x):
    xs = ia3_bypass(x, lyco.weight.to(x.dtype), -1, multiplier, False, backend)
    return linear._impl(xs)


def lora_unit(lyco, backend, base):
    w = locon_diff_weight(
        lyco.lora_down.weight, lyco.lora_up.weight, None, lyco.scale, backend
    )
    return w.reshape(lyco.shape)


def loha_unit(lyco, backend, base):
    return loha_diff_weight(
        lyco.hada_w1_b,
        lyco.hada_w1_a,
        lyco.hada_w2_b,
        lyco.hada_w2_a,
        None,
        None,
        lyco.scale,
        backend,
    )


def lokr_unit(lyco, getters, backend, base):
    return lokr_kron_weight(*kron_factors(lyco, getters), None, lyco.scale, backend)


def glora_unit(lyco, backend, base):
    return glora_diff_weight(
        base.to(lyco.a1.weight.dtype),
        lyco.a1.weight,
        lyco.a2.weight,
        lyco.b1.weight,
        lyco.b2.weight,
        lyco.scale,
        backend,
    )


def with_scalar(lyco, unit, base):
    return unit(base) * lyco.scalar


def oft_unit(lyco, backend, base):
    return diag_oft_diff_weight(base, lyco.oft_blocks, None, lyco.constraint, backend)


def oft_unit_rescaled(lyco, backend, base):
    return diag_oft_diff_weight(
        base, lyco.oft_blocks, lyco.rescale, lyco.constraint, backend
    )


def boft_unit(lyco, multiplier, backend, base):
    return boft_diff_weight(
        base, lyco.oft_blocks, None, lyco.constraint, multiplier, backend
    )


def boft_unit_rescaled(lyco, multiplier, backend, base):
    return boft_diff_weight(
        base, lyco.oft_blocks, lyco.rescale, lyco.constraint, multiplier, backend
    )


def ia3_unit(lyco, multiplier, backend, base):
    w = lyco.weight.to(base.dtype)
    return ia3_diff_weight(base, w, lyco.train_input, multiplier, True, backend)


def merged_forward(base_fn, unit, scale, lin, x):
    """y = lin(x, W + scale(dW)): W + dW summed in fp32, rounded once by `lin`."""
    w = base_fn(torch.float32)
    return lin(x, w + scale(unit(w)).float())


def decompose(lyco, on_out, backend, merged):
    """DoRA(W + dW) at multiplier 1 through the fused decompose kernel."""
    return apply_dora(merged, lyco.dora_scale, 1.0, on_out, backend)


def dora_forward(base_fn, lyco, unit, decompose, mix, lin, x):
    """y = lin(x, mix(W, decompose(W + dW)))."""
    w = base_fn(lyco.dora_scale.dtype)
    merged = w + unit(w).to(w.dtype)
    return lin(x, mix(w, decompose(merged)))
