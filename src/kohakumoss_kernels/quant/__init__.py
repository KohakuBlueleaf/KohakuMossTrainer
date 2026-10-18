"""Frozen-base quantized linear: y = x @ (q * scale[:, None])^T + bias.

`q` is int8 or float8_e4m3fn [N, K] with an fp32 per-row `scale` [N]. Backends
register a (forward, dgrad) launcher pair here; `make_quant_linear(backend)` is
the op the registry hands out: a custom op (compile-safe) whose autograd gives
the input gradient through the dgrad kernel and the bias gradient as a column
sum, saving only `q` and `scale` for backward. The base itself never receives a
gradient. The eager entry is the same op over the reference launchers.
"""

import functools
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from kohakumoss_kernels import compile_setup, registry
from kohakumoss_kernels.hardware.current import on_device

FORMATS = {"int8": torch.int8, "fp8": torch.float8_e4m3fn}

_LAUNCHERS: dict[str, tuple[Callable, Callable]] = {}


def dequantize(q: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    """The reference dequantized weight, rounded once to `dtype`."""
    return (q.to(torch.float32) * scale.float()[:, None]).to(dtype)


def _fwd_ref(x: Tensor, q: Tensor, scale: Tensor, bias: Tensor | None) -> Tensor:
    b = None if bias is None else bias.to(x.dtype)
    return F.linear(x, dequantize(q, scale, x.dtype), b)


def quant_linear_reference(
    x: Tensor, q: Tensor, scale: Tensor, bias: Tensor | None, fmt: str
) -> Tensor:
    """The eager reference: dequantize, then a dense linear (what PyTorch alone does)."""
    return _fwd_ref(x, q, scale, bias)


def _dgrad_ref(g: Tensor, q: Tensor, scale: Tensor) -> Tensor:
    return g @ dequantize(q, scale, g.dtype)


@functools.cache
def _compiled(fn: Callable) -> Callable:
    compile_setup.apply()
    return torch.compile(fn, dynamic=True)


PYTORCH_PATHS = ({"impl": "eager"}, {"impl": "compile"})


def pytorch_path(cfg: dict, which: str) -> Callable | None:
    """The dequantize+matmul call (eager or compiled) a tuner config names, or
    None for a kernel config; `which` is "fwd" or "dgrad"."""
    impl = cfg.get("impl")
    if impl is None:
        return None
    fn = _fwd_ref if which == "fwd" else _dgrad_ref
    return fn if impl == "eager" else _compiled(fn)


def register_launchers(backend: str, fwd: Callable, dgrad: Callable) -> None:
    """`fwd(x2d, q, scale, bias) -> y2d`, `dgrad(dy2d, q, scale) -> dx2d`."""
    _LAUNCHERS[backend] = (fwd, dgrad)


@torch.library.custom_op("kohakumoss::quant_fwd", mutates_args=())
def _quant_fwd(
    x: Tensor, q: Tensor, scale: Tensor, bias: Tensor | None, backend: str
) -> Tensor:
    with on_device(x.device):
        return _LAUNCHERS[backend][0](x, q, scale, bias)


@_quant_fwd.register_fake
def _(x, q, scale, bias, backend):
    return x.new_empty(x.shape[0], q.shape[0])


@torch.library.custom_op("kohakumoss::quant_dgrad", mutates_args=())
def _quant_dgrad(dy: Tensor, q: Tensor, scale: Tensor, backend: str) -> Tensor:
    with on_device(dy.device):
        return _LAUNCHERS[backend][1](dy, q, scale)


@_quant_dgrad.register_fake
def _(dy, q, scale, backend):
    return dy.new_empty(dy.shape[0], q.shape[1])


def _setup(ctx, inputs, output):
    x, q, scale, bias, backend = inputs
    ctx.save_for_backward(q, scale)
    ctx.backend = backend
    ctx.bias_dtype = None if bias is None else bias.dtype


def _backward(ctx, grad):
    q, scale = ctx.saved_tensors
    g = grad.contiguous()
    gx = None
    if ctx.needs_input_grad[0]:
        gx = _quant_dgrad(g, q, scale, ctx.backend)
    gb = None
    if ctx.bias_dtype is not None and ctx.needs_input_grad[3]:
        gb = g.float().sum(0).to(ctx.bias_dtype)
    return gx, None, None, gb, None


_quant_fwd.register_autograd(_backward, setup_context=_setup)


def make_quant_linear(backend: str) -> Callable:
    """The registry implementation of `quant_linear` for `backend`."""

    def quant_linear(
        x: Tensor, q: Tensor, scale: Tensor, bias: Tensor | None, fmt: str
    ) -> Tensor:
        lead = x.shape[:-1]
        y = _quant_fwd(x.reshape(-1, x.shape[-1]).contiguous(), q, scale, bias, backend)
        return y.reshape(*lead, q.shape[0])

    return quant_linear


def fmt_supported(fmt: str, sm: int) -> str | None:
    """Why `fmt` cannot run on `sm`, or None: int8 everywhere a backend runs, fp8
    from sm89."""
    if fmt not in FORMATS:
        return f"unknown quant format {fmt!r}"
    if fmt == "fp8" and sm < 89:
        return "fp8 weights need sm89+"
    return None


register_launchers("eager", _fwd_ref, _dgrad_ref)
quant_linear_eager = make_quant_linear("eager")
registry.register("quant_linear", "eager", quant_linear_eager)
