"""fp8 compute linear over a trainable weight: y = Q(x) Q(W)^T + bias from fp8 copies
of W that the caller rebuilds after every optimizer step. The input gradient runs
through the W^T copy (fp8) when one is given, else 16-bit dy @ W; the weight gradient
is dy^T x on 16-bit operands (x saved in its own dtype, fp32 accumulation). Uses the
`fp8_linear` launchers of each backend. See docs/internals/quant-kernels.md.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from kohakumoss_kernels import registry
from kohakumoss_kernels.backends._master import weight_grad
from kohakumoss_kernels.fp8.op import dgrad, linear_forward


@torch.library.custom_op("kohakumoss::fp8_linear_train", mutates_args=())
def _fp8_train(
    x: Tensor,
    w: Tensor,
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


@_fp8_train.register_fake
def _(x, w, q, scale, aux, q_t, scale_t, aux_t, bias, recipe, backend):
    return x.new_empty(x.shape[0], q.shape[0])


@torch.library.custom_op("kohakumoss::fp8_train_dgrad", mutates_args=())
def _fp8_train_dgrad(
    dy: Tensor,
    w: Tensor,
    q: Tensor,
    scale: Tensor,
    q_t: Tensor | None,
    scale_t: Tensor | None,
    aux_t: Tensor | None,
    recipe: str,
    backend: str,
) -> Tensor:
    return _dgrad_direct(dy, w, q, scale, q_t, scale_t, aux_t, recipe, backend)


@_fp8_train_dgrad.register_fake
def _(dy, w, q, scale, q_t, scale_t, aux_t, recipe, backend):
    return dy.new_empty(dy.shape[0], w.shape[1])


def _dgrad_direct(dy, w, q, scale, q_t, scale_t, aux_t, recipe, backend):
    """dy W through the W^T fp8 copy when one is given, else 16-bit dy @ W."""
    if q_t is None:
        return dy @ w.to(dy.dtype)
    return dgrad(dy, q, scale, q_t, scale_t, aux_t, recipe, backend)


def _setup(ctx, inputs, output):
    x, w, q, scale, aux, q_t, scale_t, aux_t, bias, recipe, backend = inputs
    keep_x = x if ctx.needs_input_grad[1] else None
    ctx.save_for_backward(keep_x, w, q, scale, q_t, scale_t, aux_t)
    ctx.recipe, ctx.backend = recipe, backend
    ctx.bias_dtype = None if bias is None else bias.dtype


def _grads(ctx, grad, input_grad: Callable):
    x, w, q, scale, q_t, scale_t, aux_t = ctx.saved_tensors
    g = grad.contiguous()
    need = ctx.needs_input_grad
    gx = gw = gb = None
    if need[0]:
        gx = input_grad(g, w, q, scale, q_t, scale_t, aux_t, ctx.recipe, ctx.backend)
    if need[1]:
        gw = weight_grad(g, x.to(g.dtype), w.dtype)
    if ctx.bias_dtype is not None and need[8]:
        gb = g.float().sum(0).to(ctx.bias_dtype)
    return gx, gw, None, None, None, None, None, None, gb, None, None


def _backward(ctx, grad):
    return _grads(ctx, grad, _fp8_train_dgrad)


_fp8_train.register_autograd(_backward, setup_context=_setup)


class _Eager(torch.autograd.Function):
    """The custom op's forward and backward without its dispatch (eager calls)."""

    @staticmethod
    def forward(ctx, x, w, q, scale, aux, q_t, scale_t, aux_t, bias, recipe, backend):
        y = linear_forward(x, q, scale, aux, bias, recipe, backend)
        inputs = (x, w, q, scale, aux, q_t, scale_t, aux_t, bias, recipe, backend)
        _setup(ctx, inputs, y)
        return y

    @staticmethod
    def backward(ctx, grad):
        return _grads(ctx, grad, _dgrad_direct)


def make_fp8_linear_train(backend: str) -> Callable:
    """The registry implementation of `fp8_linear_train` for `backend`:
    (x [..., K], w [N, K], q, scale, aux, q_t, scale_t, aux_t, bias, recipe) -> y."""

    def fp8_linear_train(x, w, q, scale, aux, q_t, scale_t, aux_t, bias, recipe):
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        fn = _fp8_train if torch.compiler.is_compiling() else _Eager.apply
        y = fn(x2, w, q, scale, aux, q_t, scale_t, aux_t, bias, recipe, backend)
        return y.reshape(*lead, q.shape[0])

    return fp8_linear_train


registry.register("fp8_linear_train", "eager", make_fp8_linear_train("eager"))
