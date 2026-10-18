"""Text-encoder linear quantization: weight-only int8 / fp8 with per-output-channel or
per-group scales, and fp8 / MXFP8 compute.

Per channel: `QuantWeight` (int8 [-127, 127] / e4m3 rows, the `quant_linear` kernels).
Per group of `group` input columns: q [out, in] with scale [out, in / group], dequantized
to the compute dtype per call. fp8_compute / mxfp8: `ComputeWeight` (pre-quantized e4m3
weights in the recipe's scale layout, activations quantized per call, the `fp8_linear`
op). See docs/internals/text-encoder.md.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kohakumoss.quant.compute import COMPUTE
from kohakumoss.quant.convert import QuantLinear, accepts
from kohakumoss.quant.weights import FP8_MAX, INT8_MAX

FMT_DTYPE = {"int8": torch.int8, "fp8": torch.float8_e4m3fn}
FMT_MAX = {"int8": INT8_MAX, "fp8": FP8_MAX}


def quantize_groups(w: Tensor, fmt: str, group: int) -> tuple[Tensor, Tensor]:
    """(q [out, in], scale [out, in / group] fp32) of a 2-D weight."""
    out_f, in_f = w.shape
    wg = w.detach().float().view(out_f, in_f // group, group)
    scale = wg.abs().amax(-1).clamp_min(1e-12) / FMT_MAX[fmt]
    x = wg / scale[..., None]
    q = torch.round(x).clamp_(-INT8_MAX, INT8_MAX) if fmt == "int8" else x.clamp_(-FP8_MAX, FP8_MAX)  # fmt: skip
    return q.view(out_f, in_f).to(FMT_DTYPE[fmt]), scale


def dequantize_groups(q: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    out_f, in_f = q.shape
    g = in_f // scale.shape[1]
    return (q.view(out_f, -1, g).float() * scale[..., None]).view(out_f, in_f).to(dtype)


class GroupQuantLinear(nn.Module):
    """y = x W^T + b with W stored per-group quantized (dequantized per call)."""

    def __init__(self, linear: nn.Linear, fmt: str, group: int):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        q, scale = quantize_groups(linear.weight, fmt, group)
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.bias = (
            None
            if linear.bias is None
            else nn.Parameter(linear.bias.detach(), requires_grad=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, dequantize_groups(self.q, self.scale, x.dtype), b)


def _make(linear, precision, group, compute_dtype, recipe, work_device):
    if group and precision not in COMPUTE:
        return GroupQuantLinear(linear, precision, group)
    return QuantLinear(linear, precision, compute_dtype, recipe, work_device)


@torch.no_grad()
def quantize_text_linears(
    module: nn.Module,
    precision: str,
    group: int,
    compute_dtype: torch.dtype,
    min_features: int = 256,
    recipe: str = "row",
    work_device: torch.device | str | None = None,
) -> dict[str, int]:
    """Replace every plain nn.Linear with min(in, out) >= `min_features` in place
    (quantizing on `work_device`); returns the layer count, dense / stored bytes and
    the layers fp8 compute could not take (`skipped`)."""
    report = {"layers": 0, "dense_bytes": 0, "quant_bytes": 0, "skipped": 0}
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is not nn.Linear:
                continue
            if min(child.in_features, child.out_features) < min_features:
                continue
            if group and child.in_features % group:
                continue
            if not accepts(precision, child.out_features, child.in_features):
                report["skipped"] += 1
                continue
            new = _make(child, precision, group, compute_dtype, recipe, work_device)
            report["layers"] += 1
            report["dense_bytes"] += child.weight.nbytes
            report["quant_bytes"] += sum(t.nbytes for t in new.buffers())
            setattr(parent, name, new)
    return report
