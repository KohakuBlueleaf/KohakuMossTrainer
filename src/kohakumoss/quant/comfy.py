"""Load ComfyUI-style quantized checkpoints (fp8 weights with scales, int8 with scales).

Recognised per-layer key sets (`<n>` is a module path):
  `<n>.weight` float8_e4m3fn + `<n>.scale_weight` (scalar or per-row)
  `<n>.weight` float8_e4m3fn / int8 + `<n>.weight_scale` (scalar or per-row)
"""

import torch
from torch import Tensor, nn

from kohakumoss.model.linear import MossLinear
from kohakumoss.quant.weights import QuantWeight

SCALE_KEYS = ("scale_weight", "weight_scale")
QUANT_DTYPES = {torch.float8_e4m3fn: "fp8", torch.int8: "int8"}
IGNORED = ("scale_input", "input_scale", "comfy_quant")


def split_quantized(
    state: dict[str, Tensor],
) -> tuple[dict[str, QuantWeight], dict[str, Tensor]]:
    """({module path: QuantWeight}, remaining dense state dict)."""
    quant: dict[str, QuantWeight] = {}
    dense = dict(state)
    for key, t in state.items():
        if not key.endswith(".weight") or t.dtype not in QUANT_DTYPES:
            continue
        path = key[: -len(".weight")]
        scale = None
        for sk in SCALE_KEYS:
            found = dense.pop(f"{path}.{sk}", None)
            scale = found if scale is None else scale
        if scale is None:
            raise KeyError(f"{key} is {t.dtype} but has no scale tensor")
        for ik in IGNORED:
            dense.pop(f"{path}.{ik}", None)
        dense.pop(key)
        rows = t.shape[0]
        scale = scale.float().reshape(-1)
        if scale.numel() == 1:
            scale = scale.expand(rows).clone()
        quant[path] = QuantWeight(t, scale, QUANT_DTYPES[t.dtype])
    return quant, dense


@torch.no_grad()
def attach_quantized(
    model: nn.Module,
    quant: dict[str, QuantWeight],
    compute_dtype: torch.dtype,
    prefix: str = "",
) -> int:
    """Attach loaded QuantWeights to the matching MossLinears; returns how many."""
    linears = {n: m for n, m in model.named_modules() if isinstance(m, MossLinear)}
    count = 0
    for path, qw in quant.items():
        name = path[len(prefix) :] if prefix and path.startswith(prefix) else path
        lin = linears.get(name)
        if lin is None:
            raise KeyError(f"no MossLinear named {name!r} for quantized key {path!r}")
        dev = lin.weight.device
        lin.weight = nn.Parameter(
            torch.empty(0, dtype=compute_dtype, device=dev), requires_grad=False
        )
        lin.set_base(qw.to(dev))
        qw.bind(compute_dtype)
        count += 1
    return count
