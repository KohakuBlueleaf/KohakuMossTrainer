"""Pre-quantized fp8-compute checkpoints: a MossBackbone state in which every fp8-compute
linear is `<n>.weight` (the e4m3 q) plus `<n>.weight_scale` (its recipe's scale), the
recipe and layer list in the safetensors metadata (`kohakumoss_fp8`). See
docs/internals/quant-kernels.md#pre-quantized-checkpoints."""

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from kohakumoss.model.linear import MossLinear
from kohakumoss.quant.compute import ComputeWeight
from kohakumoss_common.io import read_safetensors_metadata

FP8_KEY = "kohakumoss_fp8"
SCALE = ".weight_scale"


def compute_bases(model: nn.Module) -> dict[str, ComputeWeight]:
    """{module path: ComputeWeight} of every fp8-compute MossLinear."""
    return {
        path: lin.base
        for path, lin in model.named_modules()
        if isinstance(lin, MossLinear) and isinstance(lin.base, ComputeWeight)
    }


def quantized_state(
    model, keys: list[str], dtype: torch.dtype
) -> tuple[dict[str, Tensor], dict[str, str]]:
    """(state, metadata) of `model` over its base `keys`: fp8-compute layers as q and
    scale, every other tensor cast to `dtype` (rope.bands stays fp32), on the host."""
    if getattr(model, "fold_scales", None) is not None:
        raise ValueError("a folded (fp16-fix) model is exported without fp8 bases")
    bases = compute_bases(model)
    recipes = {b.recipe for b in bases.values()}
    if len(recipes) != 1:
        raise ValueError(f"one fp8 recipe per checkpoint, found {sorted(recipes)}")
    live = {**dict(model.named_buffers()), **dict(model.named_parameters())}
    state = {}
    for k in keys:
        path = k[: -len(".weight")] if k.endswith(".weight") else None
        if path in bases:
            state[k] = bases[path].q.contiguous().cpu()
            state[path + SCALE] = bases[path].scale.contiguous().cpu()
            continue
        t = live[k].detach()
        if k != "rope.bands" and t.is_floating_point():
            t = t.to(dtype)
        state[k] = t.contiguous().cpu()
    record = {"recipe": recipes.pop(), "layers": sorted(bases)}
    return state, {FP8_KEY: json.dumps(record)}


def read_record(path: str | Path) -> dict | None:
    """The fp8 record of a single-file checkpoint, if it carries one."""
    return read_meta_record(path, FP8_KEY)


def read_meta_record(path: str | Path, key: str) -> dict | None:
    """The JSON record under metadata `key` of a single safetensors file, if it carries
    one."""
    path = Path(path)
    if not path.is_file():
        return None
    meta = read_safetensors_metadata(path)
    return json.loads(meta[key]) if key in meta else None


def stored_keys(record: dict) -> tuple[set[str], set[str]]:
    """(the `.weight` keys of stored layers, their scale keys)."""
    layers = record["layers"]
    return {f"{p}.weight" for p in layers}, {p + SCALE for p in layers}


@torch.no_grad()
def attach_stored(
    model: nn.Module, view, record: dict, device, dtype: torch.dtype
) -> int:
    """Attach a ComputeWeight read from `view` to every recorded layer; returns how
    many."""
    recipe = record["recipe"]
    for path in record["layers"]:
        lin = model.get_submodule(path)
        q = view.get(f"{path}.weight").to(device)
        scale = view.get(path + SCALE).to(device)
        base = ComputeWeight.from_stored(recipe, q, scale)
        lin.weight = nn.Parameter(
            torch.empty(0, dtype=dtype, device=device), requires_grad=False
        )
        lin.set_base(base)
        base.bind(dtype, device)
    return len(record["layers"])
