"""Quantize the frozen MossLinear bases of a model in place."""

from collections.abc import Callable

import torch
from torch import nn

from kohakumoss.model.linear import MossLinear
from kohakumoss.quant.weights import QuantWeight


def default_filter(
    keep_fp32: tuple[str, ...], min_features: int = 256
) -> Callable[[str, MossLinear], bool]:
    """Quantize every MossLinear except those under a KEEP_FP32 name and small ones."""

    def accept(name: str, lin: MossLinear) -> bool:
        if any(name == k or name.startswith(k + ".") for k in keep_fp32):
            return False
        return min(lin.in_features, lin.out_features) >= min_features

    return accept


class QuantLinear(nn.Module):
    """Inference replacement of a plain nn.Linear by a quantized weight (+ bias)."""

    def __init__(self, linear: nn.Linear, fmt: str, compute_dtype: torch.dtype):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.qweight = QuantWeight.from_weight(linear.weight, fmt).to(
            linear.weight.device
        )
        self.qweight.bind(compute_dtype)
        self.bias = None
        self._impl = self._no_bias
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach(), requires_grad=False)
            self._impl = self._with_bias

    def _no_bias(self, x):
        return self.qweight.linear(x, None)

    def _with_bias(self, x):
        return self.qweight.linear(x, self.bias.to(x.dtype))

    def forward(self, x):
        return self._impl(x)


@torch.no_grad()
def quantize_torch_linears(
    module: nn.Module, fmt: str, compute_dtype: torch.dtype, min_features: int = 256
) -> int:
    """Replace every large plain nn.Linear inside `module` (e.g. a frozen text encoder)."""
    count = 0
    for name, child in list(module.named_children()):
        if type(child) is nn.Linear and min(child.in_features, child.out_features) >= (
            min_features
        ):
            setattr(module, name, QuantLinear(child, fmt, compute_dtype))
            count += 1
        else:
            count += quantize_torch_linears(child, fmt, compute_dtype, min_features)
    return count


@torch.no_grad()
def quantize_model(
    model: nn.Module,
    fmt: str,
    accept: Callable[[str, MossLinear], bool] | None = None,
    compute_dtype: torch.dtype | None = None,
) -> dict[str, int]:
    """Replace accepted MossLinear bases with QuantWeight; returns counts and bytes."""
    accept = accept or default_filter(())
    report = {"layers": 0, "dense_bytes": 0, "quant_bytes": 0}
    for name, lin in list(model.named_modules()):
        if not isinstance(lin, MossLinear) or lin.base is not None:
            continue
        if not accept(name, lin):
            continue
        w = lin.weight.detach()
        qw = QuantWeight.from_weight(w, fmt).to(w.device)
        report["layers"] += 1
        report["dense_bytes"] += w.numel() * w.element_size()
        report["quant_bytes"] += qw.nbytes()
        lin.weight = nn.Parameter(
            torch.empty(0, dtype=w.dtype, device=w.device), requires_grad=False
        )
        lin.set_base(qw)
        qw.bind(compute_dtype or w.dtype)
    return report
