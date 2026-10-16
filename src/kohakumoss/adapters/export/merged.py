"""Merged full checkpoints: base weights with every adapter ΔW (and trained norms) baked in."""

import re

import torch
from torch import Tensor, nn

from kohakumoss.adapters.delta import delta
from kohakumoss.adapters.target import CHILD, NormTarget, Target

QKV = re.compile(r"^(layers\.\d+\.proj)\.q\.weight$")


def _row(scale, like: Tensor) -> Tensor:
    s = scale if isinstance(scale, Tensor) else torch.tensor(float(scale))
    s = s.float().to(like.device).flatten()
    return s.reshape(-1, *[1] * (like.dim() - 1)) if s.numel() > 1 else s.reshape(())


def fuse_qkv(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Joint-block q/k/v -> one `proj.qkv` [3D, D] (the fused single-file layout)."""
    out = dict(state)
    for key in list(state):
        m = QKV.match(key)
        if m is None:
            continue
        stem = m.group(1)
        parts = [out.pop(f"{stem}.{p}.weight") for p in ("q", "k", "v")]
        out[f"{stem}.qkv.weight"] = torch.cat(parts, dim=0)
    return out


def merged_state_dict(
    model: nn.Module,
    targets: list[Target],
    norms: list[NormTarget],
    multiplier: float = 1.0,
    dtype: torch.dtype | None = None,
    row_scale: dict | None = None,
    fused_qkv: bool = False,
) -> dict[str, Tensor]:
    """Bare-key state dict of the backbone with adapters merged at `multiplier`.

    `row_scale` maps a module path to the per-output-row factor that undoes the
    residual fold for that layer (weight and bias rows).
    """
    marker = f".{CHILD}."
    stored = {
        path: m
        for path, m in model.named_modules()
        if getattr(m, "base", None) is not None
    }
    storage = tuple(f"{path}.base." for path in stored)
    state = {
        k: v.detach()
        for k, v in model.state_dict().items()
        if marker not in k
        and ".parametrizations." not in k
        and not k.startswith(storage)
    }
    with torch.no_grad():
        for path, linear in stored.items():
            state[f"{path}.weight"] = linear.base_weight(dtype or linear.weight.dtype)
        for t in targets:
            w = t.base_weight(torch.float32)
            dw, db = delta(t, multiplier)
            out_dtype = dtype or t.lyco.org_weight.dtype
            state[f"{t.path}.weight"] = (w + dw.to(w.device)).to(out_dtype)
            bias = t.base_bias()
            if bias is not None:
                b = bias.detach().float().to(w.device)
                if db is not None:
                    b = b + db.to(w.device)
                state[f"{t.path}.bias"] = b.to(out_dtype)
        for n in norms:
            for name, d in (("weight", n.weight_delta), ("bias", n.bias_delta)):
                if d is None:
                    continue
                orig = n.module.parametrizations[name].original.detach()
                value = orig.float() + d.w_norm.detach().float() * multiplier
                state[f"{n.path}.{name}"] = value.to(dtype or orig.dtype)
        for path, scale in (row_scale or {}).items():
            for name in ("weight", "bias"):
                key = f"{path}.{name}"
                if key in state:
                    v = state[key]
                    state[key] = (v.float() * _row(scale, v)).to(v.dtype)
    if dtype is not None:
        state = {
            k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()
        }
    return fuse_qkv(state) if fused_qkv else state
