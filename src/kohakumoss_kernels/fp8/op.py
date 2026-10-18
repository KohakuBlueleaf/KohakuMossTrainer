"""fp8 compute linear over a pre-quantized frozen base: y = Q(x) Q(W)^T + bias.

`q` [N, K] e4m3 with `scale` in the recipe's layout (`fp8.formats`); `aux` is the
vendor-layout copy of the scale (mx: the swizzled e8m0 bytes) or None. The input
gradient is Q(dy) Q(W^T)^T through the dgrad copy (`q_t` [K, N], `scale_t`, `aux_t`)
when one is stored, else the 16-bit dy @ dequant(W). Activations are quantized per
call (straight-through for the gradient). Backends register a (forward, dgrad)
launcher pair; `make_fp8_linear(backend)` is the custom op (compile-safe) the registry
hands out. See docs/internals/quant-kernels.md.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from kohakumoss_kernels import registry
from kohakumoss_kernels.fp8.formats import dequantize, quantize
from kohakumoss_kernels.hardware.current import on_device

_LAUNCHERS: dict[str, tuple[Callable, Callable]] = {}


def fwd_reference(x, q, scale, aux, bias, recipe):
    """The eager reference forward: quantize x, dequantize both operands to x's dtype,
    then F.linear."""
    xq, xs = quantize(x, recipe, "act")
    a = dequantize(xq, xs, recipe, x.dtype)
    b = None if bias is None else bias.to(x.dtype)
    return F.linear(a, dequantize(q, scale, recipe, x.dtype), b)


def dgrad_reference(dy, q, scale, q_t, scale_t, aux_t, recipe):
    """The eager reference input gradient: fp8 through the dgrad copy, else 16-bit."""
    if q_t is None:
        return dy @ dequantize(q, scale, recipe, dy.dtype)
    gq, gs = quantize(dy, recipe, "act")
    a = dequantize(gq, gs, recipe, dy.dtype)
    return F.linear(a, dequantize(q_t, scale_t, recipe, dy.dtype))


def register_launchers(backend: str, fwd: Callable, dgrad: Callable) -> None:
    """`fwd(x2d, q, scale, aux, bias, recipe) -> y2d`,
    `dgrad(dy2d, q, scale, q_t, scale_t, aux_t, recipe) -> dx2d`."""
    _LAUNCHERS[backend] = (fwd, dgrad)


def linear_forward(x, q, scale, aux, bias, recipe, backend):
    """The forward of `fp8_linear` through `backend`'s launcher, outside autograd
    (`x` [M, K] -> [M, N])."""
    with on_device(x.device):
        return _LAUNCHERS[backend][0](x, q, scale, aux, bias, recipe)


def dgrad(dy, q, scale, q_t, scale_t, aux_t, recipe, backend):
    """The input gradient of `fp8_linear` through `backend`'s launcher, outside
    autograd (`dy` [M, N] -> [M, K]); callers that hold the weight parts elsewhere."""
    with on_device(dy.device):
        return _LAUNCHERS[backend][1](dy, q, scale, q_t, scale_t, aux_t, recipe)


@torch.library.custom_op("kohakumoss::fp8_dgrad", mutates_args=())
def _fp8_dgrad(
    dy: Tensor,
    q: Tensor,
    scale: Tensor,
    q_t: Tensor | None,
    scale_t: Tensor | None,
    aux_t: Tensor | None,
    recipe: str,
    backend: str,
) -> Tensor:
    return dgrad(dy, q, scale, q_t, scale_t, aux_t, recipe, backend)


@_fp8_dgrad.register_fake
def _(dy, q, scale, q_t, scale_t, aux_t, recipe, backend):
    return dy.new_empty(dy.shape[0], q.shape[1])


@torch.library.custom_op("kohakumoss::fp8_linear", mutates_args=())
def _fp8_linear(
    x: Tensor,
    q: Tensor,
    scale: Tensor,
    aux: Tensor | None,
    q_t: Tensor | None,
    scale_t: Tensor | None,
    aux_t: Tensor | None,
    bias: Tensor | None,
    recipe: str,
    backend: str,
) -> Tensor:
    return linear_forward(x, q, scale, aux, bias, recipe, backend)


@_fp8_linear.register_fake
def _(x, q, scale, aux, q_t, scale_t, aux_t, bias, recipe, backend):
    return x.new_empty(x.shape[0], q.shape[0])


def _setup(ctx, inputs, output):
    x, q, scale, aux, q_t, scale_t, aux_t, bias, recipe, backend = inputs
    ctx.save_for_backward(q, scale, q_t, scale_t, aux_t)
    ctx.recipe, ctx.backend = recipe, backend
    ctx.bias_dtype = None if bias is None else bias.dtype


def _grads(ctx, grad, input_grad: Callable):
    q, scale, q_t, scale_t, aux_t = ctx.saved_tensors
    g = grad.contiguous()
    gx = None
    if ctx.needs_input_grad[0]:
        gx = input_grad(g, q, scale, q_t, scale_t, aux_t, ctx.recipe, ctx.backend)
    gb = None
    if ctx.bias_dtype is not None and ctx.needs_input_grad[7]:
        gb = g.float().sum(0).to(ctx.bias_dtype)
    return gx, None, None, None, None, None, None, gb, None, None


def _backward(ctx, grad):
    return _grads(ctx, grad, _fp8_dgrad)


_fp8_linear.register_autograd(_backward, setup_context=_setup)


class _Eager(torch.autograd.Function):
    """The custom op's forward and backward without its dispatch (eager calls)."""

    @staticmethod
    def forward(ctx, x, q, scale, aux, q_t, scale_t, aux_t, bias, recipe, backend):
        y = linear_forward(x, q, scale, aux, bias, recipe, backend)
        _setup(ctx, (x, q, scale, aux, q_t, scale_t, aux_t, bias, recipe, backend), y)
        return y

    @staticmethod
    def backward(ctx, grad):
        return _grads(ctx, grad, dgrad)


def make_fp8_linear(backend: str) -> Callable:
    """The registry implementation of `fp8_linear` for `backend`:
    (x [..., K], q, scale, aux, q_t, scale_t, aux_t, bias, recipe) -> y [..., N]
    (the custom op under torch.compile, the same autograd function otherwise)."""

    def fp8_linear(x, q, scale, aux, q_t, scale_t, aux_t, bias, recipe):
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        fn = _fp8_linear if torch.compiler.is_compiling() else _Eager.apply
        y = fn(x2, q, scale, aux, q_t, scale_t, aux_t, bias, recipe, backend)
        return y.reshape(*lead, q.shape[0])

    return fp8_linear


def make_fp8_linear_infer(backend: str) -> Callable:
    """`fp8_linear` without autograd for frozen, gradient-free callers (the forward
    launcher called directly; the custom op under torch.compile)."""

    def fp8_linear_infer(x, q, scale, aux, q_t, scale_t, aux_t, bias, recipe):
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        if torch.compiler.is_compiling():
            y = _fp8_linear(x2, q, scale, aux, q_t, scale_t, aux_t, bias, recipe, backend)  # fmt: skip
        else:
            with torch.no_grad():
                y = linear_forward(x2, q, scale, aux, bias, recipe, backend)
        return y.reshape(*lead, q.shape[0])

    return fp8_linear_infer


register_launchers("eager", fwd_reference, dgrad_reference)
registry.register("fp8_linear", "eager", make_fp8_linear("eager"))
registry.register("fp8_linear_infer", "eager", make_fp8_linear_infer("eager"))
