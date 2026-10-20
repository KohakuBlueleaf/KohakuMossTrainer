"""Quantize the frozen MossLinear bases of a model in place: weight-only (int8 / fp8,
`QuantWeight`) or fp8 compute (fp8_compute / mxfp8, `ComputeWeight`)."""

from collections.abc import Callable

import torch
from torch import nn

from kohakumoss.model.linear import MossLinear
from kohakumoss.quant.compute import (
    COMPUTE,
    ComputeCache,
    ComputeWeight,
    eligible,
    recipe_of,
)
from kohakumoss.quant.source import (
    DEFERRED,
    STORED,
    SourceWeights,
    deferred_weight,
    log_sources,
    source_weight,
    tally,
)
from kohakumoss.quant.weights import QuantWeight


def make_base(
    w: torch.Tensor, fmt: str, recipe: str = "row", dgrad: bool = False
) -> QuantWeight | ComputeWeight:
    """The quantized base of weight `w` under precision `fmt` (`recipe` and `dgrad`
    apply to fp8_compute / mxfp8)."""
    if fmt in COMPUTE:
        return ComputeWeight.from_weight(w, recipe_of(fmt, recipe), dgrad)
    return QuantWeight.from_weight(w, fmt)


def accepts(fmt: str, out_features: int, in_features: int) -> bool:
    """Whether a [N, K] weight can take `fmt` (fp8 compute needs aligned shapes)."""
    return fmt not in COMPUTE or eligible(out_features, in_features) is None


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
    """Inference replacement of a plain nn.Linear by a quantized weight (+ bias),
    quantized from `values` when given (its stored values, `source_weight`), else from
    the linear's weight."""

    def __init__(
        self,
        linear: nn.Linear,
        fmt: str,
        compute_dtype: torch.dtype,
        recipe: str = "row",
        work_device: torch.device | str | None = None,
        values: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        dev = linear.weight.device
        w = linear.weight.detach() if values is None else values
        self.qweight = make_base(w.to(work_device or dev), fmt, recipe).to(dev)
        self.qweight.bind(compute_dtype)
        self.bias = None
        self._impl = QuantLinear._no_bias
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach(), requires_grad=False)
            self._impl = QuantLinear._with_bias

    @classmethod
    def from_base(
        cls,
        in_features: int,
        out_features: int,
        base: QuantWeight | ComputeWeight,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
    ) -> "QuantLinear":
        """A QuantLinear over an already-quantized base (a pre-quantized file)."""
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.in_features, self.out_features = in_features, out_features
        self.qweight = base
        self.qweight.bind(compute_dtype)
        self.bias = None
        self._impl = QuantLinear._no_bias
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
            self._impl = QuantLinear._with_bias
        return self

    def _no_bias(self, x):
        return self.qweight.linear(x, None)

    def _with_bias(self, x):
        return self.qweight.linear(x, self.bias.to(x.dtype))

    def forward(self, x):
        return self._impl(self, x)


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
            if not accepts(fmt, child.out_features, child.in_features):
                continue
            setattr(module, name, QuantLinear(child, fmt, compute_dtype))
            count += 1
        else:
            count += quantize_torch_linears(child, fmt, compute_dtype, min_features)
    return count


@torch.no_grad()
def attach_compute(
    model: nn.Module,
    fmt: str,
    accept: Callable[[str, MossLinear], bool],
    compute_dtype: torch.dtype,
    recipe: str = "row",
    dgrad: bool = True,
) -> list[MossLinear]:
    """Trainable fp8 compute (fp8_compute / mxfp8): a ComputeCache, quantized from the
    current weight, on every accepted dense MossLinear of an fp8-compute shape; returns
    those linears (their caches need `refresh` after every optimizer step)."""
    out = []
    for name, lin in list(model.named_modules()):
        if not isinstance(lin, MossLinear) or lin.base is not None:
            continue
        if not accept(name, lin) or not accepts(fmt, lin.out_features, lin.in_features):
            continue
        cache = ComputeCache(recipe_of(fmt, recipe), dgrad)
        cache.refresh(lin.weight)
        cache.bind(compute_dtype, lin.weight.device)
        lin.set_compute(cache)
        out.append(lin)
    return out


def quantizable(
    model: nn.Module, fmt: str, accept: Callable[[str, MossLinear], bool]
) -> tuple[list[tuple[str, MossLinear]], int]:
    """(the dense MossLinears `quantize_model` replaces under `fmt` and `accept`, by
    module path; how many accepted ones fp8 compute's shapes refuse)."""
    out, skipped = [], 0
    for name, lin in model.named_modules():
        if not isinstance(lin, MossLinear) or lin.base is not None:
            continue
        if not accept(name, lin):
            continue
        if accepts(fmt, lin.out_features, lin.in_features):
            out.append((name, lin))
        else:
            skipped += 1
    return out, skipped


def unread_problems(model: nn.Module) -> list[str]:
    """Weights the load left unread (`deferred_weights`, for a quantizer reading the
    checkpoint) that are still not quantized."""
    left = sorted(getattr(model, "deferred_weights", {}))
    if not left:
        return []
    return [
        f"{len(left)} weights the load left unread for quantization are not "
        f"quantized: {left[:3]}"
    ]


@torch.no_grad()
def quantize_model(
    model: nn.Module,
    fmt: str,
    accept: Callable[[str, MossLinear], bool] | None = None,
    compute_dtype: torch.dtype | None = None,
    recipe: str = "row",
    dgrad: bool = False,
    work_device: torch.device | str | None = None,
    source: SourceWeights | None = None,
    scales: dict[str, torch.Tensor] | None = None,
) -> dict[str, int]:
    """Replace accepted MossLinear bases with QuantWeight (int8 / fp8) or ComputeWeight
    (fp8_compute / mxfp8; `recipe`, `dgrad` copy), quantizing on `work_device`
    (default: where the weight is) from each weight's stored values in `source` (times
    its fold rows in `scales`, by module path) where `source_weight` takes them, and
    always for a weight the load left unread (`model.deferred_weights`: name -> its
    home device), the tally logged once; returns counts and bytes (`skipped`: accepted
    layers whose shape fp8 compute cannot take, `from_source`: layers quantized from
    stored values, `sources`: layers per `source_weight` kind)."""
    accept = accept or default_filter(())
    linears, skipped = quantizable(model, fmt, accept)
    report = {"layers": 0, "dense_bytes": 0, "quant_bytes": 0, "skipped": skipped}
    report["from_source"] = 0
    sources: dict[str, int] = {}
    scales = scales or {}
    deferred = getattr(model, "deferred_weights", {})
    for name, lin in linears:
        w = lin.weight.detach()
        key = f"{name}.weight"
        home = deferred.pop(key, None)
        if home is not None:
            work = work_device or home
            values = deferred_weight(source, key, w.dtype, scales.get(name), work)
            kind = DEFERRED
        else:
            home = w.device
            values, kind = source_weight(
                w, source, key, scales.get(name), work_device or w.device
            )
        tally(sources, kind)
        qw = make_base(values, fmt, recipe, dgrad).to(home)
        report["layers"] += 1
        report["dense_bytes"] += w.nbytes
        report["quant_bytes"] += qw.nbytes()
        lin.weight = nn.Parameter(
            torch.empty(0, dtype=w.dtype, device=home), requires_grad=False
        )
        lin.set_base(qw)
        qw.bind(compute_dtype or w.dtype)
    report["from_source"] = sources.get(STORED, 0) + sources.get(DEFERRED, 0)
    report["sources"] = sources
    if source is not None:
        log_sources(f"base {fmt}", sources)
    return report
