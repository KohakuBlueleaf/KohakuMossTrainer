"""Build LyCORIS modules for the targets of a preset (LyCORIS targeting semantics).

A quantized / streamed MossLinear keeps its empty `weight` placeholder: LyCORIS sizes
every module from in_features / out_features and, seeing a non-native linear, runs it
in bypass. The state LyCORIS would derive from the base weight at construction (the
DoRA magnitude, the T-LoRA SVD, the OFT rescale) is built here from one transient
copy of each layer's base, so no dense weight is ever held. See
docs/internals/adapter-binding.md#quantized-and-streamed-bases.
"""

import re
from types import SimpleNamespace

import torch
from lycoris.modules.tlora import TLoraModule
from lycoris.wrapper import LycorisNetwork
from torch import nn

from kohakumoss.adapters.build.config import AdapterConfig
from kohakumoss.adapters.build.presets import resolve_preset
from kohakumoss.adapters.build.target import NormTarget, Target, weak_org
from kohakumoss.adapters.train import norms
from kohakumoss.adapters.train.tlora import AXES, axis_of


def lora_name_of(prefix: str, path: str) -> str:
    """LyCORIS / kohya key of a module path: `{prefix}_{path with . -> _}`."""
    return f"{prefix}_{path}".replace(".", "_")


def _paths(model: nn.Module) -> dict[int, str]:
    return {id(m): name for name, m in model.named_modules()}


def transient_base(linear: nn.Module) -> torch.Tensor:
    """One fp32 host copy of the layer's base weight (dequantized / gathered from
    its storage), for construction-time statistics only."""
    fn = getattr(linear, "base_weight", None)
    w = fn(torch.float32) if fn is not None else linear.weight.float()
    return w.detach().cpu()


class Deferred:
    """The construction-time state taken out of the LyCORIS kwargs (it would read the
    base weight) and built per layer by `restore`."""

    def __init__(self, kwargs: dict, algo: str):
        self.wd = bool(kwargs.pop("weight_decompose", False))
        self.wd_on_out = bool(kwargs.pop("wd_on_out", True))
        self.rescaled = bool(kwargs.pop("rescaled", False))
        self.svd = algo == "tlora" and bool(kwargs.get("use_data_init", True))
        if self.svd:
            kwargs["use_data_init"] = False

    def restore(self, lyco: nn.Module, linear: nn.Module) -> None:
        if not (self.wd or self.rescaled or self.svd):
            return
        w = transient_base(linear)
        if self.wd:
            norm = w.norm(dim=1 if self.wd_on_out else 0, keepdim=True)
            lyco.wd, lyco.wd_on_out, lyco.dora_norm_dims = True, self.wd_on_out, 1
            lyco.dora_scale = nn.Parameter(norm)
        if self.rescaled:
            lyco.rescaled = True
            lyco.rescale = nn.Parameter(torch.ones(w.shape[0], 1))
        if self.svd and isinstance(lyco, TLoraModule):
            svd_init(lyco, w)
        del w


@torch.no_grad()
def svd_init(lyco: TLoraModule, w: torch.Tensor) -> None:
    """LyCORIS's data-dependent T-LoRA init from base weight `w`, and the frozen base
    factors re-taken from it."""
    lyco.use_data_init = True
    proxy = SimpleNamespace(weight=SimpleNamespace(data=w))
    out_dim, in_dim = w.shape
    lyco._initialize_from_svd(proxy, in_dim, out_dim, in_dim)
    lyco.base_q.copy_(lyco.q_layer.weight)
    lyco.base_p.copy_(lyco.p_layer.weight)
    lyco.base_lambda.copy_(lyco.lambda_layer)


def _target(path: str, lyco: nn.Module, linear: nn.Module) -> Target:
    """A Target; T-LoRA modules read the mask group of their token axis."""
    t = Target(path=path, lora_name=lyco.lora_name, linear=linear, lyco=lyco)
    if t.record.per_token_mask:
        t.axis = axis_of(path)
        lyco.mask_group_id = AXES[t.axis]
    return t


def build_targets(
    model: nn.Module, config: AdapterConfig
) -> tuple[list[Target], list[NormTarget]]:
    """LyCORIS modules for every matched linear, and the norms that train directly."""
    preset, norm_patterns = resolve_preset(config.preset, config.prefix)
    LycorisNetwork.apply_preset(preset)
    kwargs = config.module_kwargs()
    deferred = Deferred(kwargs, config.lycoris_algo)
    net = LycorisNetwork(
        model,
        multiplier=config.multiplier,
        lora_dim=config.dim,
        alpha=config.alpha,
        dropout=config.dropout,
        rank_dropout=config.rank_dropout,
        module_dropout=0.0,
        network_module=config.lycoris_algo,
        train_norm=False,
        warn_on_unmatched=True,
        **kwargs,
    )
    loras, net.loras = net.loras, []
    paths = _paths(model)
    targets = []
    for lyco in loras:
        linear = lyco.org_module[0]
        if "forward" in vars(linear):
            raise RuntimeError(f"{paths[id(linear)]} already carries an adapter")
        weak_org(lyco, linear)
        deferred.restore(lyco, linear)
        targets.append(_target(paths[id(linear)], lyco, linear))
    trained = build_norms(model, norm_patterns, config) if config.train_norms else []
    return targets, trained


def build_norms(model: nn.Module, patterns: list[str], config: AdapterConfig):
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
                lora_name=lora_name_of(config.prefix, path),
                module=module,
                weight_delta=norms.attach(module, "weight"),
                bias_delta=norms.attach(module, "bias") if has_bias else None,
            )
        )
    return out
