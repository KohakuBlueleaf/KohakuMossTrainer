"""Build LyCORIS modules for the targets of a preset (LyCORIS targeting semantics)."""

import re

import torch
from lycoris.wrapper import LycorisNetwork
from torch import nn

from kohakumoss.adapters import norms
from kohakumoss.adapters.presets import resolve_preset
from kohakumoss.adapters.spec import AdapterSpec
from kohakumoss.adapters.target import NormTarget, Target
from kohakumoss.adapters.tlora import AXES, axis_of


def lora_name_of(prefix: str, path: str) -> str:
    """LyCORIS / kohya key of a module path: `{prefix}_{path with . -> _}`."""
    return f"{prefix}_{path}".replace(".", "_")


def _paths(model: nn.Module) -> dict[int, str]:
    return {id(m): name for name, m in model.named_modules()}


class _DenseWeight:
    """Context exposing a dense `weight` on one linear with quantized/streamed storage."""

    def __init__(self, linear: nn.Module):
        self.linear = linear
        self.saved = None

    def __enter__(self):
        self.saved = self.linear._parameters.get("weight")
        dense = self.linear.base.materialize().detach().cpu()
        self.linear._parameters["weight"] = nn.Parameter(dense, requires_grad=False)
        return self.linear

    def __exit__(self, *exc):
        self.linear._parameters["weight"] = self.saved
        return False


_LAZY: dict[type, type] = {}


def _lazy_weight(self):
    return self.__dict__["_lazy_scope"].dense(self)


def _lazy_class(cls: type) -> type:
    """`cls` with a `weight` property served by the lazy scope (same class name, so
    LyCORIS's class-name rules still match)."""
    if cls not in _LAZY:
        _LAZY[cls] = type(cls.__name__, (cls,), {"weight": property(_lazy_weight)})
    return _LAZY[cls]


class _LazyDense:
    """Within the context, reading a stored linear's `weight` materialises its base
    on the host on first access; only the most recently read one stays resident."""

    def __init__(self, linears: list[nn.Module]):
        self.linears = [(m, type(m)) for m in linears]
        self.current = None
        self.value = None

    def dense(self, linear: nn.Module) -> nn.Parameter:
        if self.current is not linear:
            self.current, self.value = linear, None
            dense = linear.base.materialize().detach().cpu()
            self.value = nn.Parameter(dense, requires_grad=False)
        return self.value

    def __enter__(self):
        for m, cls in self.linears:
            m.__dict__["_lazy_scope"] = self
            m.__class__ = _lazy_class(cls)
        return self

    def __exit__(self, *exc):
        for m, cls in self.linears:
            m.__class__ = cls
            m.__dict__.pop("_lazy_scope", None)
        self.current = self.value = None
        return False


def _stored_linears(model: nn.Module) -> list[nn.Module]:
    return [
        m
        for m in model.modules()
        if isinstance(m, nn.Linear) and getattr(m, "base", None) is not None
    ]


def _target(path: str, lyco: nn.Module, linear: nn.Module) -> Target:
    """A Target; T-LoRA modules read the mask group of their token axis."""
    t = Target(path=path, lora_name=lyco.lora_name, linear=linear, lyco=lyco)
    if t.record.token_axis:
        t.axis = axis_of(path)
        lyco.mask_group_id = AXES[t.axis]
    return t


def build_targets(
    model: nn.Module, spec: AdapterSpec
) -> tuple[list[Target], list[NormTarget]]:
    """LyCORIS modules for every matched linear, and the norms that train directly."""
    preset, norm_patterns = resolve_preset(spec.preset, spec.prefix)
    LycorisNetwork.apply_preset(preset)
    with _LazyDense(_stored_linears(model)):
        net = LycorisNetwork(
            model,
            multiplier=spec.multiplier,
            lora_dim=spec.dim,
            alpha=spec.alpha,
            dropout=spec.dropout,
            rank_dropout=spec.rank_dropout,
            module_dropout=0.0,
            network_module=spec.lycoris_algo,
            train_norm=False,
            warn_on_unmatched=True,
            **spec.module_kwargs(),
        )
    paths = _paths(model)
    targets = []
    for lyco in net.loras:
        linear = lyco.org_module[0]
        if "forward" in vars(linear):
            raise RuntimeError(f"{paths[id(linear)]} already carries an adapter")
        targets.append(_target(paths[id(linear)], lyco, linear))
    norms_ = build_norms(model, norm_patterns, spec) if spec.train_norms else []
    return targets, norms_


def build_norms(model: nn.Module, patterns: list[str], spec: AdapterSpec):
    """Norm modules whose path matches any pattern; their weights train in fp32."""
    compiled = [re.compile(p) for p in patterns]
    found = []
    for path, module in model.named_modules():
        if not any(c.match(path) for c in compiled):
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.dim() == 1:
            found.append((path, module))
    out = []
    for path, module in found:
        bias = getattr(module, "bias", None)
        has_bias = isinstance(bias, torch.Tensor) and bias.dim() == 1
        out.append(
            NormTarget(
                path=path,
                lora_name=lora_name_of(spec.prefix, path),
                module=module,
                weight_delta=norms.attach(module, "weight"),
                bias_delta=norms.attach(module, "bias") if has_bias else None,
            )
        )
    return out
