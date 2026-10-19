"""Rotated frozen-base quantized linear (ConvRot): y = quant_linear(x . R, q, scale, b)
with R = blockdiag(H_group) over K (`kohakumoss_kernels.hadamard`).

The stored weight is the rotated one, W R ~ q * scale, so y = x W^T and the input
gradient is dx = (dy (q * scale)) R. Backends register a (fwd_rot, dgrad_rot) launcher
pair; `compose` builds the pair from a backend's plain quant launchers and a rotation
(the separate-pass path every backend has). `make_quant_linear_rotated(backend)` is the
registry's `quant_linear_rotated`: a custom op whose autograd gives the input gradient
through `dgrad_rot` and the bias gradient as a column sum, saving only `q` and `scale`.
See docs/internals/quant-kernels.md#rotated-int8-linear.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from kohakumoss_kernels import registry
from kohakumoss_kernels.hadamard import group_hadamard_eager, stages
from kohakumoss_kernels.hardware.current import on_device
from kohakumoss_kernels.quant import _dgrad_ref, _fwd_ref

__all__ = [
    "compose",
    "group_supported",
    "make_quant_linear_rotated",
    "quant_linear_rotated_eager",
    "quant_linear_rotated_reference",
    "register_rotated_launchers",
]

_LAUNCHERS: dict[str, tuple[Callable, Callable]] = {}


def quant_linear_rotated_reference(
    x: Tensor, q: Tensor, scale: Tensor, bias: Tensor | None, fmt: str, group: int
) -> Tensor:
    """The eager reference: rotate x, dequantize, then a dense linear."""
    return _fwd_ref(group_hadamard_eager(x, group), q, scale, bias)


def compose(fwd: Callable, dgrad: Callable, rotate: Callable) -> tuple:
    """(fwd_rot, dgrad_rot) of plain launchers `fwd(x2d, q, scale, bias)`,
    `dgrad(dy2d, q, scale)` and `rotate(t2d, group)`: one rotation pass each."""

    def fwd_rot(x, q, scale, bias, group):
        return fwd(rotate(x, group), q, scale, bias)

    def dgrad_rot(dy, q, scale, group):
        return rotate(dgrad(dy, q, scale), group)

    return fwd_rot, dgrad_rot


def register_rotated_launchers(backend: str, fwd_rot: Callable, dgrad_rot: Callable):
    """`fwd_rot(x2d, q, scale, bias, group) -> y2d`, `dgrad_rot(dy2d, q, scale, group)
    -> dx2d` (dx already rotated back)."""
    _LAUNCHERS[backend] = (fwd_rot, dgrad_rot)


@torch.library.custom_op("kohakumoss::quant_fwd_rot", mutates_args=())
def _fwd_rot(
    x: Tensor, q: Tensor, scale: Tensor, bias: Tensor | None, group: int, backend: str
) -> Tensor:
    with on_device(x.device):
        return _LAUNCHERS[backend][0](x, q, scale, bias, group)


@_fwd_rot.register_fake
def _(x, q, scale, bias, group, backend):
    return x.new_empty(x.shape[0], q.shape[0])


@torch.library.custom_op("kohakumoss::quant_dgrad_rot", mutates_args=())
def _dgrad_rot(
    dy: Tensor, q: Tensor, scale: Tensor, group: int, backend: str
) -> Tensor:
    with on_device(dy.device):
        return _LAUNCHERS[backend][1](dy, q, scale, group)


@_dgrad_rot.register_fake
def _(dy, q, scale, group, backend):
    return dy.new_empty(dy.shape[0], q.shape[1])


def _setup(ctx, inputs, output):
    x, q, scale, bias, group, backend = inputs
    ctx.save_for_backward(q, scale)
    ctx.group, ctx.backend = group, backend
    ctx.bias_dtype = None if bias is None else bias.dtype


def _backward(ctx, grad):
    q, scale = ctx.saved_tensors
    g = grad.contiguous()
    gx = None
    if ctx.needs_input_grad[0]:
        gx = _dgrad_rot(g, q, scale, ctx.group, ctx.backend)
    gb = None
    if ctx.bias_dtype is not None and ctx.needs_input_grad[3]:
        gb = g.float().sum(0).to(ctx.bias_dtype)
    return gx, None, None, gb, None, None


_fwd_rot.register_autograd(_backward, setup_context=_setup)


def make_quant_linear_rotated(backend: str) -> Callable:
    """The registry implementation of `quant_linear_rotated` for `backend`."""

    def quant_linear_rotated(
        x: Tensor, q: Tensor, scale: Tensor, bias: Tensor | None, fmt: str, group: int
    ) -> Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        y = _fwd_rot(x2, q, scale, bias, group, backend)
        return y.reshape(*lead, q.shape[0])

    return quant_linear_rotated


def group_supported(query) -> str | None:
    """Why the query's static `group` cannot rotate, or None (no group given: None)."""
    group = query.get("group")
    if group is None:
        return None
    try:
        stages(group)
    except ValueError as err:
        return str(err)
    return None


register_rotated_launchers(
    "eager", *compose(_fwd_ref, _dgrad_ref, group_hadamard_eager)
)
quant_linear_rotated_eager = make_quant_linear_rotated("eager")
registry.register(
    "quant_linear_rotated",
    "eager",
    quant_linear_rotated_eager,
    supports=group_supported,
)
