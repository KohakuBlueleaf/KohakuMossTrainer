"""Bound forwards over LyCORIS module methods and the bound dW builders, built once per
bind as closures; `bind.select` picks one per layer. Training-mode variants are re-bound
by the adapter set's train()/eval() hook."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from kohakumoss.adapters.bind.bodies import ia3_scale

F32 = torch.float32


def module_bypass(linear, lyco, multiplier: float) -> Callable:
    """x -> base(x) + the LyCORIS module's bypass delta rounded to x's dtype."""

    def forward(x: Tensor) -> Tensor:
        return linear._impl(x) + lyco.bypass_forward_diff(x, multiplier).to(x.dtype)

    return forward


def module_rotation(lyco, multiplier: float) -> Callable:
    """x -> the LyCORIS module's bypass forward: a rotation of the base output, which
    the module computes through its own `org_forward` (one base forward)."""

    def forward(x: Tensor) -> Tensor:
        return lyco.bypass_forward(x, multiplier)

    return forward


def glora_bypass(linear, lyco, multiplier: float) -> Callable:
    """x -> base(x) + W A x + B x: LyCORIS's GLoRA bypass delta runs W through the base
    forward, so the base bias it adds once more is taken back out."""
    if linear.bias is None:
        return module_bypass(linear, lyco, multiplier)

    def forward(x: Tensor) -> Tensor:
        y = linear._impl(x)
        dy = lyco.bypass_forward_diff(x, multiplier).to(y.dtype)
        return y + dy - linear.bias.to(y.dtype)

    return forward


def ia3_out(linear, lyco, multiplier: float) -> Callable:
    """x -> base(x) (w m + 1)."""

    def forward(x: Tensor) -> Tensor:
        return ia3_scale(linear._impl(x), lyco.weight.to(x.dtype), multiplier)

    return forward


def ia3_in(linear, lyco, multiplier: float) -> Callable:
    """x -> base(x (w m + 1))."""

    def forward(x: Tensor) -> Tensor:
        return linear._impl(ia3_scale(x, lyco.weight.to(x.dtype), multiplier))

    return forward


def dylora_train(linear, lyco, multiplier: float, draws) -> Callable:
    """x -> base(x) + x down^T up^T over the rank prefix drawn for this step (no dW)."""

    def forward(x: Tensor) -> Tensor:
        down, up, scale = lyco.get_weight(draws.rank)
        mid = F.linear(x, down.to(x.dtype))
        y = F.linear(mid, up.to(x.dtype))
        return linear._impl(x) + y * (scale * multiplier)

    return forward


def delta_bypass(linear, shape, delta_fn: Callable, dense: Callable) -> Callable:
    """x -> base(x) + x dW^T with dW from the bound builder (base storage untouched)
    and `dense` the layer's matmul."""

    def forward(x: Tensor) -> Tensor:
        return linear._impl(x) + dense(x, delta_fn()[0].reshape(shape))

    return forward


def eager_merged(base_fn, shape, delta_fn: Callable, lin: Callable) -> Callable:
    """x -> lin(x, W + dW) with dW from the bound builder."""

    def forward(x: Tensor) -> Tensor:
        w = base_fn(F32)
        return lin(x, w + delta_fn()[0].reshape(shape))

    return forward


def eager_dora(base_fn, unit_fn, decompose, mix, lin) -> Callable:
    """x -> lin(x, mix(W, decompose(W + dW))) with the bound epilogue variant."""

    def forward(x: Tensor) -> Tensor:
        w = base_fn(F32)
        return lin(x, mix(w, decompose(w + unit_fn())))

    return forward


def module_dropout(linear, draws, inner: Callable) -> Callable:
    """Training only: the adapter is skipped for this step when its draw says so."""

    def forward(x: Tensor) -> Tensor:
        if draws.skip:
            return linear._impl(x)
        return inner(x)

    return forward
