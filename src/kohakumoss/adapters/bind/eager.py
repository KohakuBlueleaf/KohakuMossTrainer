"""Bound forwards over LyCORIS module methods and the bound ΔW builders.

Used for dropout, layers outside a fused kernel's scope and the eager backends. Every
variant is its own function; `bind.select` picks one per layer, and training-mode
variants are re-bound by the adapter set's train()/eval() hook.
"""

import torch
import torch.nn.functional as F
from lycoris.modules.ia3 import _ia3_scale


def module_bypass(lyco, multiplier, x):
    """LyCORIS bypass_forward: base output plus the adapter path, in x's dtype."""
    return lyco.bypass_forward(x, multiplier).to(x.dtype)


def module_forward(lyco, x):
    """The LyCORIS module's own forward (used for `full`, which owns the weight)."""
    return lyco.forward(x)


def ia3_out(linear, lyco, multiplier, x):
    return _ia3_scale(linear._impl(x), lyco.weight.to(x.dtype), multiplier, False)


def ia3_in(linear, lyco, multiplier, x):
    return linear._impl(_ia3_scale(x, lyco.weight.to(x.dtype), multiplier, False))


def dylora_train(linear, lyco, multiplier, x):
    """base(x) + x ΔW^T over a random rank prefix (LyCORIS's DyLoRA training draw)."""
    dw = lyco.get_diff_weight(multiplier)[0].reshape(lyco.shape)
    return linear._impl(x) + F.linear(x, dw.to(x.dtype))


def delta_bypass(linear, shape, delta_fn, x):
    """base(x) + x ΔW^T with ΔW from the bound builder (base storage untouched)."""
    return linear._impl(x) + F.linear(x, delta_fn()[0].reshape(shape).to(x.dtype))


def eager_merged(base_fn, shape, delta_fn, lin, x):
    """lin(x, W + ΔW) with ΔW from the bound builder."""
    w = base_fn(torch.float32)
    return lin(x, w + delta_fn()[0].reshape(shape))


def eager_dora(base_fn, unit_fn, decompose, mix, lin, x):
    """lin(x, mix(W, decompose(W + ΔW))) with the bound epilogue variant."""
    w = base_fn(torch.float32)
    return lin(x, mix(w, decompose(w + unit_fn())))


def module_dropout(linear, p, inner, x):
    """Training only: skip the adapter for the whole call with probability p."""
    if torch.rand(()) < p:
        return linear._impl(x)
    return inner(x)
