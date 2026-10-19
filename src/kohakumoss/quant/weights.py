"""Quantized frozen base weights (per-output-row scales) as MossLinear base storage.

int8: q = round(w / s) clamped to [-127, 127], s = max|w_row| / 127.
fp8 (e4m3fn): q = fp8(w / s), s = max|w_row| / 448.
`linear` uses a registered `quant_linear` kernel when one serves the device, else
the eager quant op (dequantize + matmul, saving only `q` / `scale` for backward); the
choice is made once per weight, not per call.
"""

from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss_kernels import registry
from kohakumoss_kernels.quant import quant_linear_eager

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


class QuantWeight(nn.Module):
    """BaseWeight holding `q` and `scale` as buffers (they follow `.to(device)`)."""

    def __init__(self, q: Tensor, scale: Tensor, fmt: str):
        super().__init__()
        self.fmt = fmt
        self.register_buffer("q", q)
        self.register_buffer("scale", scale.float())
        self.shape = q.shape
        self._linear = _dequant_linear

    @classmethod
    def from_weight(cls, w: Tensor, fmt: str) -> "QuantWeight":
        q, scale = quantize(w, fmt)
        return cls(q, scale, fmt)

    def weight(self, dtype: torch.dtype) -> Tensor:
        return dequantize(self.q, self.scale, dtype)

    def materialize(self) -> Tensor:
        return dequantize(self.q, self.scale, torch.float32)

    def bind(
        self, dtype: torch.dtype, device: torch.device | str | None = None
    ) -> None:
        """Bind `linear` for (device, dtype) once: the `quant_linear` kernel when one
        serves, else eager dequantization."""
        device = self.q.device if device is None else torch.device(device)
        try:
            kernel = registry.resolve("quant_linear", device, dtype, fmt=self.fmt)
        except (registry.NoImplementationError, KeyError):
            self._linear = _dequant_linear
            return
        self._linear = partial(_kernel_linear, kernel)

    def linear(self, x: Tensor, bias: Tensor | None) -> Tensor:
        """x W^T (+ bias) through the bound implementation."""
        return self._linear(self, x, bias)

    def nbytes(self) -> int:
        return self.q.nbytes + self.scale.numel() * 4


def _dequant_linear(qw: QuantWeight, x: Tensor, bias: Tensor | None) -> Tensor:
    return quant_linear_eager(x, qw.q, qw.scale, bias, qw.fmt)


def _kernel_linear(kernel, qw: QuantWeight, x: Tensor, bias: Tensor | None) -> Tensor:
    return kernel(x, qw.q, qw.scale, bias, qw.fmt)
