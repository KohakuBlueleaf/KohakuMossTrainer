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
from kohakumoss.quant.source import (
    STORED,
    SourceWeights,
    log_sources,
    source_weight,
    tally,
)
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
    """y = x W^T + b with W stored per-group quantized (dequantized per call), from
    `values` when given (the stored values, `source_weight`), else the linear's weight.
    """

    def __init__(
        self, linear: nn.Linear, fmt: str, group: int, values: Tensor | None = None
    ):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        w = linear.weight if values is None else values
        q, scale = quantize_groups(w, fmt, group)
        q, scale = q.to(linear.weight.device), scale.to(linear.weight.device)
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


def _make(linear, precision, group, compute_dtype, recipe, work_device, values):
    if group and precision not in COMPUTE:
        return GroupQuantLinear(linear, precision, group, values)
    return QuantLinear(linear, precision, compute_dtype, recipe, work_device, values)


def linear_fate(
    module: nn.Module, precision: str, group: int, min_features: int = 256
) -> str | None:
    """What `quantize_text_linears` does with `module`: "quantize", "skip" (a plain
    linear fp8 compute cannot take) or None (left as it is)."""
    if type(module) is not nn.Linear:
        return None
    if min(module.in_features, module.out_features) < min_features:
        return None
    if group and module.in_features % group:
        return None
    if not accepts(precision, module.out_features, module.in_features):
        return "skip"
    return "quantize"


def quantizes(
    module: nn.Module, precision: str, group: int, min_features: int = 256
) -> bool:
    """Whether `quantize_text_linears` would replace any linear inside `module`."""
    return any(
        linear_fate(m, precision, group, min_features) == "quantize"
        for m in module.modules()
    )


@torch.no_grad()
def quantize_text_linears(
    module: nn.Module,
    precision: str,
    group: int,
    compute_dtype: torch.dtype,
    min_features: int = 256,
    recipe: str = "row",
    work_device: torch.device | str | None = None,
    source: SourceWeights | None = None,
    quiet: bool = False,
) -> dict[str, int]:
    """Replace every plain nn.Linear with min(in, out) >= `min_features` in place
    (quantizing on `work_device`, from each weight's stored values in `source`, by
    weight name under `module`, where `source_weight` takes them, the tally logged
    once unless `quiet`); each dense linear is freed as it is replaced. Returns the
    layer count, dense / stored bytes, the layers fp8 compute could not take
    (`skipped`), those quantized from stored values (`from_source`) and the layers per
    `source_weight` kind (`sources`)."""
    report = {"layers": 0, "dense_bytes": 0, "quant_bytes": 0, "skipped": 0}
    sources: dict[str, int] = {}
    parents = [(p, m) for p, m in module.named_modules() if type(m) is not nn.Linear]
    for path, parent in parents:
        names = [n for n, c in parent.named_children() if type(c) is nn.Linear]
        for name in names:
            child = getattr(parent, name)
            fate = linear_fate(child, precision, group, min_features)
            if fate == "skip":
                report["skipped"] += 1
            if fate != "quantize":
                continue
            weight = f"{path}.{name}.weight" if path else f"{name}.weight"
            values, kind = source_weight(
                child.weight, source, weight, work=work_device or child.weight.device
            )
            tally(sources, kind)
            new = _make(
                child, precision, group, compute_dtype, recipe, work_device, values
            )
            report["layers"] += 1
            report["dense_bytes"] += child.weight.nbytes
            report["quant_bytes"] += sum(t.nbytes for t in new.buffers())
            setattr(parent, name, new)
            del child, values, new
    report["from_source"] = sources.get(STORED, 0)
    report["sources"] = sources
    if source is not None and not quiet:
        log_sources(f"text encoder {precision}", sources)
    return report
