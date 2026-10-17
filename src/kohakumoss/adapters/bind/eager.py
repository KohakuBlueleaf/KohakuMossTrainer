"""Bound forwards over LyCORIS module methods and the bound dW builders.

Each variant is its own function; `bind.select` picks one per layer. Training-mode
variants are re-bound by the adapter set's train()/eval() hook.
"""

import torch
from lycoris.modules.ia3 import ia3_scale


def module_bypass(lyco, multiplier, x):
    """LyCORIS bypass_forward: base output plus the adapter path, in x's dtype."""
    return lyco.bypass_forward(x, multiplier).to(x.dtype)


def module_forward(lyco, x):
    """The LyCORIS module's own forward (used for `full`, which owns the weight)."""
    return lyco.forward(x)


def ia3_out(linear, lyco, multiplier, x):
    return ia3_scale(linear._impl(x), lyco.weight.to(x.dtype), multiplier, False)


def ia3_in(linear, lyco, multiplier, x):
    return linear._impl(ia3_scale(x, lyco.weight.to(x.dtype), multiplier, False))


def dylora_train(linear, lyco, multiplier, draws, dense, x):
    """base(x) + x dW^T over the rank prefix drawn for this step."""
    dw = lyco.get_diff_weight(multiplier, rank=draws.rank)[0].reshape(lyco.shape)
    return linear._impl(x) + dense(x, dw)


def delta_bypass(linear, shape, delta_fn, dense, x):
    """base(x) + x dW^T with dW from the bound builder (base storage untouched) and
    `dense` the layer's matmul."""
    return linear._impl(x) + dense(x, delta_fn()[0].reshape(shape))


def eager_merged(base_fn, shape, delta_fn, lin, x):
    """lin(x, W + dW) with dW from the bound builder."""
    w = base_fn(torch.float32)
    return lin(x, w + delta_fn()[0].reshape(shape))


def eager_dora(base_fn, unit_fn, decompose, mix, lin, x):
    """lin(x, mix(W, decompose(W + dW))) with the bound epilogue variant."""
    w = base_fn(torch.float32)
    return lin(x, mix(w, decompose(w + unit_fn())))


def module_dropout(linear, draws, inner, x):
    """Training only: the adapter is skipped for this step when its draw says so."""
    if draws.skip:
        return linear._impl(x)
    return inner(x)
