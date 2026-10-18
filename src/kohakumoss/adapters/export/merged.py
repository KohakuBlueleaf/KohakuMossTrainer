"""Merged checkpoints: the backbone's export state with every adapter dW (and trained
norm delta) baked in, through `kohakumoss.model.io.save.export_state`."""

import torch
from torch import Tensor

from kohakumoss.adapters.build.target import NormTarget, Target
from kohakumoss.model.io.save import export_state


def merged_weights(
    targets: list[Target], norms: list[NormTarget], multiplier: float
) -> dict[str, Tensor]:
    """{state key: W + m * dW} (and biases, norm weights) in the model's own space."""
    out = {}
    with torch.no_grad():
        for t in targets:
            w = t.base_weight(torch.float32)
            dw, db = t.delta(multiplier)
            out[f"{t.path}.weight"] = w + dw.to(w.device)
            bias = t.base_bias()
            if bias is not None:
                b = bias.detach().float().to(w.device)
                out[f"{t.path}.bias"] = b if db is None else b + db.to(w.device)
            elif db is not None:
                raise ValueError(f"{t.path}: a bias delta on a layer without a bias")
        for n in norms:
            for name, d in (("weight", n.weight_delta), ("bias", n.bias_delta)):
                if d is None:
                    continue
                orig = n.module.parametrizations[name].original.detach().float()
                out[f"{n.path}.{name}"] = orig + d.w_norm.detach().float() * multiplier
    return out


def merged_state_dict(
    model,
    targets: list[Target],
    norms: list[NormTarget],
    multiplier: float = 1.0,
    dtype: torch.dtype | None = None,
    fused_qkv: bool = False,
) -> dict[str, Tensor]:
    """Bare-key, unfolded, cast export state of `model` with the adapters merged."""
    weights = merged_weights(targets, norms, multiplier)
    return export_state(model, dtype, fused_qkv, weights=weights)
