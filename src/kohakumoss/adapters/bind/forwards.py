"""Forward wrappers a layer's kernel ops plug into, built once per bind as closures:
x -> the adapted layer's output. `layer` is a weak reference to the layer, read per
call; the base forward is its `_impl` (a bias change re-binds it). See
docs/internals/adapter-binding.md#forward-wrappers and
docs/internals/reference-cycles.md#adapters."""

from collections.abc import Callable

import torch
from torch import Tensor

F32 = torch.float32


def plus_base(layer, diff: Callable) -> Callable:
    """x -> base(x) + diff(x)."""

    def forward(x: Tensor) -> Tensor:
        lin = layer()
        return lin._impl(lin, x) + diff(x)

    return forward


def plus_base_scalar(layer, lyco, diff: Callable) -> Callable:
    """x -> base(x) + diff(x) * the module's trainable scalar."""

    def forward(x: Tensor) -> Tensor:
        lin = layer()
        return lin._impl(lin, x) + diff(x) * lyco.scalar.to(x.dtype)

    return forward


def with_base(layer, op: Callable) -> Callable:
    """x -> op((x, base(x))) on (tokens, channels) views: base + dy, the add inside
    the op."""

    def forward(x: Tensor) -> Tensor:
        lin = layer()
        y = lin._impl(lin, x)
        out = op((x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])))
        return out.reshape(y.shape)

    return forward


def rotated(layer, rotate: Callable) -> Callable:
    """x -> rotate(y), y = base(x): the rotated base output, y + dy inside the op."""

    def forward(x: Tensor) -> Tensor:
        lin = layer()
        y = lin._impl(lin, x)
        return rotate(y).to(y.dtype)

    return forward


def plus_rotation(layer, delta: Callable) -> Callable:
    """x -> y + delta(y), y = base(x): a rotation whose dy is a function of y."""

    def forward(x: Tensor) -> Tensor:
        lin = layer()
        y = lin._impl(lin, x)
        return y + delta(y).to(y.dtype)

    return forward


def scaled_output(layer, scale: Callable) -> Callable:
    """x -> scale(base(x)): a channel scale of the base output."""

    def forward(x: Tensor) -> Tensor:
        lin = layer()
        return scale(lin._impl(lin, x))

    return forward


def scaled_input(layer, scale: Callable) -> Callable:
    """x -> base(scale(x)): a channel scale of the input."""

    def forward(x: Tensor) -> Tensor:
        lin = layer()
        return lin._impl(lin, scale(x))

    return forward


def merged(base_fn: Callable, unit: Callable, lin: Callable) -> Callable:
    """x -> lin(x, W + unit(W)): W + dW summed in fp32, rounded once by `lin`."""

    def forward(x: Tensor) -> Tensor:
        w = base_fn(F32)
        return lin(x, w + unit(w).float())

    return forward


def merged_scaled(base_fn: Callable, unit: Callable, m: float, lin: Callable):
    """x -> lin(x, W + m unit(W))."""

    def forward(x: Tensor) -> Tensor:
        w = base_fn(F32)
        return lin(x, w + (unit(w) * m).float())

    return forward


def merged_with_base(base_fn: Callable, merge: Callable, lin: Callable) -> Callable:
    """x -> lin(x, merge(W)): W + dW from one op on the fp32 base, rounded by `lin`."""

    def forward(x: Tensor) -> Tensor:
        return lin(x, merge(base_fn(F32)))

    return forward


def dora(base_fn, lyco, unit, decompose, mix, lin) -> Callable:
    """x -> lin(x, mix(W, decompose(W + unit(W))))."""

    def forward(x: Tensor) -> Tensor:
        w = base_fn(lyco.dora_scale.dtype)
        merged_w = w + unit(w).to(w.dtype)
        return lin(x, mix(w, decompose(merged_w)))

    return forward


def dora_with_delta(base_fn, lyco, unit, decompose, mix, lin) -> Callable:
    """x -> lin(x, mix(W, decompose((W, unit(W))))): W + dW read inside the op."""

    def forward(x: Tensor) -> Tensor:
        w = base_fn(lyco.dora_scale.dtype)
        return lin(x, mix(w, decompose((w, unit(w).to(w.dtype)))))

    return forward
