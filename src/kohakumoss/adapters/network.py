"""Build LyCORIS modules for the targets of a preset (LyCORIS targeting semantics)."""

import re

import torch
from lycoris.wrapper import LycorisNetwork
from torch import nn

from kohakumoss.adapters import norms
from kohakumoss.adapters.delta import resolve
from kohakumoss.adapters.presets import resolve_preset
from kohakumoss.adapters.spec import AdapterSpec
from kohakumoss.adapters.target import NormTarget, Target, kind_of


def lora_name_of(prefix: str, path: str) -> str:
    """LyCORIS / kohya key of a module path: `{prefix}_{path with . -> _}`."""
    return f"{prefix}_{path}".replace(".", "_")


DENSE_INIT = ("tlora", "full", "diag-oft", "boft")


def _paths(model: nn.Module) -> dict[int, str]:
    return {id(m): name for name, m in model.named_modules()}


class _DenseWeight:
    """Context exposing a dense `weight` on a linear with quantized/streamed storage."""

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


def build_targets(
    model: nn.Module, spec: AdapterSpec
) -> tuple[list[Target], list[NormTarget]]:
    """LyCORIS modules for every matched linear, and the norms that train directly."""
    preset, norm_patterns = resolve_preset(spec.preset, spec.prefix)
    LycorisNetwork.apply_preset(preset)
    needs_dense = (
        spec.dora
        or spec.lycoris_algo in DENSE_INIT
        or bool(preset.get("module_algo_map") or preset.get("name_algo_map"))
    )
    linears = [
        m
        for m in model.modules()
        if needs_dense
        and isinstance(m, nn.Linear)
        and getattr(m, "base", None) is not None
    ]
    contexts = [_DenseWeight(m) for m in linears]
    for ctx in contexts:
        ctx.__enter__()
    try:
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
    finally:
        for ctx in contexts:
            ctx.__exit__(None, None, None)
    paths = _paths(model)
    targets = []
    for lyco in net.loras:
        linear = lyco.org_module[0]
        if "forward" in vars(linear):
            raise RuntimeError(f"{paths[id(linear)]} already carries an adapter")
        targets.append(
            Target(
                path=paths[id(linear)],
                lora_name=lyco.lora_name,
                linear=linear,
                lyco=lyco,
                kind=kind_of(lyco),
                wd=bool(getattr(lyco, "wd", False)),
                wd_on_out=bool(getattr(lyco, "wd_on_out", True)),
            )
        )
        resolve(targets[-1])
    norms = build_norms(model, norm_patterns, spec) if spec.train_norms else []
    return targets, norms


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
