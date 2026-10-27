"""int8 compute linear over a frozen int8 base: y = Q(x R) q^T * s_x s_w + bias.

`q` [N, K] int8 with fp32 per-row `scale` [N] stores W R (R = blockdiag(H_group) over K,
ConvRot; `group` 0: no rotation). The forward rotates x, quantizes it per row to int8
(`int8.formats.quantize_rows`) and runs an int8 GEMM with int32 accumulation, both
scales applied to the fp32 accumulator. The input gradient (`dgrad`):

    int8   Q(dy * scale) q (int32) times dy's row scales, rotated back
    16bit  (dy (q * scale)) R, the weight-only input gradient

`make_int8_linear(backend)` is the custom op (compile-safe) over a backend's (forward,
dgrad) launchers, saving only `q` and `scale`. See
docs/internals/quant-kernels.md#int8-compute.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from kohakumoss_kernels import registry
from kohakumoss_kernels.hadamard import group_hadamard_eager
from kohakumoss_kernels.hardware.current import on_device
from kohakumoss_kernels.int8.formats import col_major, int_mm, quantize_rows
from kohakumoss_kernels.quant import dequantize
from kohakumoss_kernels.quant.rotated import group_supported

DGRAD = ("int8", "16bit")

_LAUNCHERS: dict[str, tuple[Callable, Callable]] = {}


def _rotate(t: Tensor, group: int) -> Tensor:
    return group_hadamard_eager(t, group) if group else t


def fwd_reference(
    x: Tensor, q: Tensor, scale: Tensor, bias: Tensor | None, group: int
) -> Tensor:
    """The eager reference forward: rotate x (in x's dtype), quantize per row, int8
    GEMM, (acc * s_x) * s_w (+ bias) in fp32, rounded once to x's dtype."""
    xq, xs = quantize_rows(_rotate(x, group))
    y = int_mm(xq, q.t()).float() * xs[:, None] * scale.float()[None, :]
    if bias is not None:
        y = y + bias.float()
    return y.to(x.dtype)


def dgrad_reference(
    dy: Tensor, q: Tensor, scale: Tensor, group: int, dgrad: str
) -> Tensor:
    """The eager reference input gradient under `dgrad` (int8 | 16bit)."""
    if dgrad == "16bit":
        return _rotate(dy @ dequantize(q, scale, dy.dtype), group)
    gq, gs = quantize_rows(dy, scale)
    dx = (int_mm(gq, col_major(q)).float() * gs[:, None]).to(dy.dtype)
    return _rotate(dx, group)


def register_launchers(backend: str, fwd: Callable, dgrad: Callable) -> None:
    """`fwd(x2d, q, scale, bias, group) -> y2d`,
    `dgrad(dy2d, q, scale, group, dgrad) -> dx2d` (dx rotated back)."""
    _LAUNCHERS[backend] = (fwd, dgrad)


def linear_forward(x, q, scale, bias, group, backend):
    """The forward of `int8_linear` through `backend`'s launcher, outside autograd
    (`x` [M, K] -> [M, N])."""
    with on_device(x.device):
        return _LAUNCHERS[backend][0](x, q, scale, bias, group)


def input_grad(dy, q, scale, group, dgrad, backend):
    """The input gradient of `int8_linear` through `backend`'s launcher, outside
    autograd (`dy` [M, N] -> [M, K]); callers holding the weight parts elsewhere."""
    with on_device(dy.device):
        return _LAUNCHERS[backend][1](dy, q, scale, group, dgrad)


@torch.library.custom_op("kohakumoss::int8_dgrad", mutates_args=())
def _int8_dgrad(
    dy: Tensor, q: Tensor, scale: Tensor, group: int, dgrad: str, backend: str
) -> Tensor:
    return input_grad(dy, q, scale, group, dgrad, backend)


@_int8_dgrad.register_fake
def _(dy, q, scale, group, dgrad, backend):
    return dy.new_empty(dy.shape[0], q.shape[1])


@torch.library.custom_op("kohakumoss::int8_linear", mutates_args=())
def _int8_linear(
    x: Tensor,
    q: Tensor,
    scale: Tensor,
    bias: Tensor | None,
    group: int,
    dgrad: str,
    backend: str,
) -> Tensor:
    return linear_forward(x, q, scale, bias, group, backend)


@_int8_linear.register_fake
def _(x, q, scale, bias, group, dgrad, backend):
    return x.new_empty(x.shape[0], q.shape[0])


def _setup(ctx, inputs, output):
    x, q, scale, bias, group, dgrad, backend = inputs
    ctx.save_for_backward(q, scale)
    ctx.group, ctx.dgrad, ctx.backend = group, dgrad, backend
    ctx.bias_dtype = None if bias is None else bias.dtype


def _grads(ctx, grad, grad_fn: Callable):
    q, scale = ctx.saved_tensors
    g = grad.contiguous()
    gx = None
    if ctx.needs_input_grad[0]:
        gx = grad_fn(g, q, scale, ctx.group, ctx.dgrad, ctx.backend)
    gb = None
    if ctx.bias_dtype is not None and ctx.needs_input_grad[3]:
        gb = g.float().sum(0).to(ctx.bias_dtype)
    return gx, None, None, gb, None, None, None


def _backward(ctx, grad):
    return _grads(ctx, grad, _int8_dgrad)


_int8_linear.register_autograd(_backward, setup_context=_setup)


class _Eager(torch.autograd.Function):
    """The custom op's forward and backward without its dispatch (eager calls)."""

    @staticmethod
    def forward(ctx, x, q, scale, bias, group, dgrad, backend):
        y = linear_forward(x, q, scale, bias, group, backend)
        _setup(ctx, (x, q, scale, bias, group, dgrad, backend), y)
        return y

    @staticmethod
    def backward(ctx, grad):
        return _grads(ctx, grad, input_grad)


def make_int8_linear(backend: str) -> Callable:
    """The registry implementation of `int8_linear` for `backend`:
    (x [..., K], q, scale, bias, group, dgrad) -> y [..., N] (the custom op under
    torch.compile, the same autograd function otherwise; the forward launcher alone
    when no gradient is needed)."""

    def int8_linear(x, q, scale, bias, group, dgrad):
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        if torch.compiler.is_compiling():
            y = _int8_linear(x2, q, scale, bias, group, dgrad, backend)
        elif torch.is_grad_enabled() and (
            x.requires_grad or (bias is not None and bias.requires_grad)
        ):
            y = _Eager.apply(x2, q, scale, bias, group, dgrad, backend)
        else:
            y = linear_forward(x2, q, scale, bias, group, backend)
        return y.reshape(*lead, q.shape[0])

    return int8_linear


def dgrad_supported(query) -> str | None:
    """Why the query's `dgrad` is not a mode, or None (none given: None)."""
    mode = query.get("dgrad")
    if mode is None or mode in DGRAD:
        return None
    return f"int8 dgrad {mode!r} not in {DGRAD}"


def int8_supported(query) -> str | None:
    """Why the query cannot run (`group` 0 or a rotation group, `dgrad` a mode), or
    None."""
    rotated = query.get("group") not in (None, 0)
    return (group_supported(query) if rotated else None) or dgrad_supported(query)


register_launchers("eager", fwd_reference, dgrad_reference)
int8_linear_eager = make_int8_linear("eager")
registry.register("int8_linear", "eager", int8_linear_eager, supports=int8_supported)
