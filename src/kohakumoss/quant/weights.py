"""Quantized frozen base weights (per-output-row scales) as MossLinear base storage.

int8: q = round(w / s) clamped to [-127, 127], s = max|w_row| / 127.
fp8 (e4m3fn): q = fp8(w / s), s = max|w_row| / 448.
`rotation` g > 0 (ConvRot, a ComfyUI file): the stored weight is W R ~ q * s with
R = blockdiag(H_g) over K, so y = (x R) (q * s)^T and W = (q * s) R.
`linear` uses a registered `quant_linear` (`quant_linear_rotated`) kernel when one
serves the device, else the eager op (dequantize + matmul, saving only `q` / `scale`
for backward); with `int8_dgrad` set (int8 compute, `set_int8_compute`) an int8 base
off the host runs the `int8_linear` op instead; on the host the eager composition runs
as plain ops over the dense weight `dense` (built per layer per pass, `build_dense`) or
one dequantized per call; the choice is made once per weight, not per call.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kohakumoss_kernels import registry
from kohakumoss_kernels.hadamard import group_hadamard_eager
from kohakumoss_kernels.quant import quant_linear_eager
from kohakumoss_kernels.quant.rotated import quant_linear_rotated_eager

FP8_MAX = 448.0
INT8_MAX = 127.0
FORMATS = ("int8", "fp8")


def quantize(w: Tensor, fmt: str) -> tuple[Tensor, Tensor]:
    """(q, scale[out]) of a 2-D weight; scale is fp32."""
    wf = w.detach().float()
    amax = wf.abs().amax(dim=1).clamp_min(1e-12)
    match fmt:
        case "int8":
            scale = amax / INT8_MAX
            q = torch.round(wf / scale[:, None]).clamp_(-INT8_MAX, INT8_MAX)
            return q.to(torch.int8), scale
        case "fp8":
            scale = amax / FP8_MAX
            q = (wf / scale[:, None]).clamp_(-FP8_MAX, FP8_MAX)
            return q.to(torch.float8_e4m3fn), scale
        case _:
            raise ValueError(f"quant format {fmt!r} not in {FORMATS}")


def dequantize(q: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    return (q.to(torch.float32) * scale.float()[:, None]).to(dtype)


def unrotated(w: Tensor, rotation: int, dtype: torch.dtype) -> Tensor:
    """W from a stored rotated fp32 weight W R (rows times R, R its own inverse),
    rounded once to `dtype`; `rotation` 0 is a plain cast."""
    if not rotation:
        return w.to(dtype)
    return group_hadamard_eager(w.float(), rotation).to(dtype)


class QuantWeight(nn.Module):
    """BaseWeight holding `q` and `scale` as buffers (they follow `.to(device)`);
    `rotation` is the ConvRot group of a rotated stored weight (0: none);
    `int8_dgrad` is None (weight-only) or the input-gradient mode of int8 compute."""

    def __init__(
        self,
        q: Tensor,
        scale: Tensor,
        fmt: str,
        rotation: int = 0,
        int8_dgrad: str | None = None,
    ):
        super().__init__()
        self.fmt = fmt
        self.rotation = int(rotation)
        self.int8_dgrad = int8_dgrad
        self.register_buffer("q", q)
        self.register_buffer("scale", scale.float())
        self.shape = q.shape
        self.dense: Tensor | None = None
        self._linear = _eager_for(self.rotation)

    @classmethod
    def from_weight(cls, w: Tensor, fmt: str) -> "QuantWeight":
        q, scale = quantize(w, fmt)
        return cls(q, scale, fmt)

    def weight(self, dtype: torch.dtype) -> Tensor:
        """W in `dtype` (the rotation undone)."""
        if not self.rotation:
            return dequantize(self.q, self.scale, dtype)
        w = dequantize(self.q, self.scale, torch.float32)
        return unrotated(w, self.rotation, dtype)

    def materialize(self) -> Tensor:
        return self.weight(torch.float32)

    def bind(
        self, dtype: torch.dtype, device: torch.device | str | None = None
    ) -> None:
        """Bind `linear` for (device, dtype) once: on the host the plain-op composition
        (`_host_linear`, its GEMMs visible to `host_gemm`); int8 compute the
        `int8_linear` op; else the `quant_linear` (rotated: `quant_linear_rotated`)
        kernel when one serves, else the eager op."""
        device = self.q.device if device is None else torch.device(device)
        if device.type == "cpu":
            self._linear = _host_linear
            return
        if self.int8_dgrad is not None:
            kernel = registry.resolve(
                "int8_linear", device, dtype, group=self.rotation, dgrad=self.int8_dgrad
            )
            self._linear = partial(_int8_linear, kernel)
            return
        name = "quant_linear_rotated" if self.rotation else "quant_linear"
        query = {"fmt": self.fmt}
        if self.rotation:
            query["group"] = self.rotation
        try:
            kernel = registry.resolve(name, device, dtype, **query)
        except (registry.NoImplementationError, KeyError):
            self._linear = _eager_for(self.rotation)
            return
        run = _rotated_linear if self.rotation else _kernel_linear
        self._linear = partial(run, kernel)

    def linear(self, x: Tensor, bias: Tensor | None) -> Tensor:
        """x W^T (+ bias) through the bound implementation."""
        return self._linear(self, x, bias)

    def build_dense(self, dtype: torch.dtype) -> None:
        """Hold `dense`, the stored weight (rotated as stored) dequantized to `dtype`,
        for the host linear's calls until `drop_dense`."""
        self.dense = dequantize(self.q, self.scale, dtype)

    def drop_dense(self) -> None:
        self.dense = None

    def nbytes(self) -> int:
        return self.q.nbytes + self.scale.numel() * 4


def _eager_for(rotation: int):
    return _rotated_eager if rotation else _dequant_linear


def _dequant_linear(qw: QuantWeight, x: Tensor, bias: Tensor | None) -> Tensor:
    return quant_linear_eager(x, qw.q, qw.scale, bias, qw.fmt)


def _rotated_eager(qw: QuantWeight, x: Tensor, bias: Tensor | None) -> Tensor:
    return quant_linear_rotated_eager(x, qw.q, qw.scale, bias, qw.fmt, qw.rotation)


def _rotate(t: Tensor, group: int) -> Tensor:
    return group_hadamard_eager(t, group) if group else t


class _HostLinear(torch.autograd.Function):
    """The eager composition as plain ops, which a dispatch mode sees (x rotated, then
    a dense linear over `dense` when it is in x's dtype, else the weight dequantized
    for this call), saving only `q` / `scale` for the input gradient."""

    @staticmethod
    def forward(ctx, x, q, scale, bias, group: int, dense):
        w = dense
        if w is None or w.dtype != x.dtype:
            w = dequantize(q, scale, x.dtype)
        b = None if bias is None else bias.to(x.dtype)
        ctx.save_for_backward(q, scale)
        ctx.group = group
        ctx.bias_dtype = None if bias is None else bias.dtype
        return F.linear(_rotate(x, group), w, b)

    @staticmethod
    def backward(ctx, grad):
        q, scale = ctx.saved_tensors
        gx = gb = None
        if ctx.needs_input_grad[0]:
            gx = _rotate(grad @ dequantize(q, scale, grad.dtype), ctx.group)
        if ctx.bias_dtype is not None and ctx.needs_input_grad[3]:
            gb = grad.float().sum(0).to(ctx.bias_dtype)
        return gx, None, None, gb, None, None


def _host_linear(qw: QuantWeight, x: Tensor, bias: Tensor | None) -> Tensor:
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    y = _HostLinear.apply(x2, qw.q, qw.scale, bias, qw.rotation, qw.dense)
    return y.reshape(*lead, qw.q.shape[0])


def _kernel_linear(kernel, qw: QuantWeight, x: Tensor, bias: Tensor | None) -> Tensor:
    return kernel(x, qw.q, qw.scale, bias, qw.fmt)


def _rotated_linear(kernel, qw: QuantWeight, x: Tensor, bias: Tensor | None) -> Tensor:
    return kernel(x, qw.q, qw.scale, bias, qw.fmt, qw.rotation)


def _int8_linear(kernel, qw: QuantWeight, x: Tensor, bias: Tensor | None) -> Tensor:
    return kernel(x, qw.q, qw.scale, bias, qw.rotation, qw.int8_dgrad)
