"""In-memory ComfyUI quantized states (the formats and their parsing are
`quant.comfy_format`): split a state dict into bases and dense tensors, attach bases to
a model's MossLinears."""

import torch
from torch import Tensor, nn

from kohakumoss.model.linear import MossLinear
from kohakumoss.quant.comfy_format import comfy_layers, layer_base
from kohakumoss.quant.w4a8 import W4A8Weight
from kohakumoss.quant.weights import QuantWeight


def split_quantized(
    state: dict[str, Tensor],
) -> tuple[dict[str, QuantWeight | W4A8Weight], dict[str, Tensor]]:
    """({module path: base}, remaining dense state dict) of an in-memory state;
    ValueError for a record of an unknown format."""
    dtypes = {k: v.dtype for k, v in state.items()}
    layers = comfy_layers(list(state), state.__getitem__, dtypes.get)
    quant = {
        path: layer_base(layer, state.__getitem__) for path, layer in layers.items()
    }
    owned = {k for layer in layers.values() for k in layer.keys}
    return quant, {k: v for k, v in state.items() if k not in owned}


@torch.no_grad()
def attach_quantized(
    model: nn.Module,
    quant: dict[str, QuantWeight | W4A8Weight],
    compute_dtype: torch.dtype,
    prefix: str = "",
) -> int:
    """Attach loaded bases to the matching MossLinears; returns how many."""
    linears = {n: m for n, m in model.named_modules() if isinstance(m, MossLinear)}
    count = 0
    for path, base in quant.items():
        name = path[len(prefix) :] if prefix and path.startswith(prefix) else path
        lin = linears.get(name)
        if lin is None:
            raise KeyError(f"no MossLinear named {name!r} for quantized key {path!r}")
        dev = lin.weight.device
        lin.weight = nn.Parameter(
            torch.empty(0, dtype=compute_dtype, device=dev), requires_grad=False
        )
        lin.set_base(base.to(dev))
        base.bind(compute_dtype)
        count += 1
    return count
