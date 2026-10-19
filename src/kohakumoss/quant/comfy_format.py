"""ComfyUI quantized checkpoints: a quantized layer `<n>` stores its codes as
`<n>.weight`, its scales, and `<n>.comfy_quant`, a uint8 JSON record of the format:

| format | tensors | base |
| --- | --- | --- |
| `int8_tensorwise` | weight I8 [N, K], weight_scale F32 [N, 1] or a scalar | QuantWeight int8 |
| `float8_e4m3fn` | weight F8 [N, K], weight_scale (or scale_weight) | QuantWeight fp8 |
| `asym_w4a8_int8` | weight I8 packed, weight_s_rel, weight_s_channel, weight_codebook | W4A8Weight |

`convrot` / `convrot_groupsize` give the rotation of a ConvRot file. A record-less
`<n>.weight` in int8 / fp8 next to `scale_weight` / `weight_scale` is read the same way
(no rotation). Imports no model module (the checkpoint loader reads it). See
docs/guides/weights.md#quantized-files.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kohakumoss.quant.w4a8 import W4A8Weight
from kohakumoss.quant.weights import QuantWeight

RECORD = "comfy_quant"
FORMATS = {"int8_tensorwise": "int8", "float8_e4m3fn": "fp8", "asym_w4a8_int8": "w4a8"}
SCALE_KEYS = ("weight_scale", "scale_weight")
W4A8_PARTS = ("s_rel", "s_channel", "codebook", "correction")
IGNORED = ("scale_input", "input_scale")
CODE_DTYPES = {torch.float8_e4m3fn: "fp8", torch.int8: "int8"}


@dataclass(frozen=True)
class ComfyLayer:
    """One quantized layer of a file: module `path`, format (`int8` | `fp8` | `w4a8`),
    ConvRot `rotation` (0: none), W4A8 `group`, and every file key it owns."""

    path: str
    fmt: str
    rotation: int
    group: int
    keys: tuple[str, ...]

    @property
    def layout(self) -> str:
        """`<fmt>`, plus `-g<group>` (w4a8) and `-convrot<rotation>`: the stored layout
        a cache key records."""
        parts = [self.fmt]
        if self.group:
            parts.append(f"g{self.group}")
        if self.rotation:
            parts.append(f"convrot{self.rotation}")
        return "-".join(parts)


def read_record(t: Tensor) -> dict:
    """The JSON record of a `comfy_quant` uint8 tensor."""
    return json.loads(t.to(torch.uint8).numpy().tobytes().decode("utf-8"))


def _owned(path: str, keys: set[str], fmt: str) -> tuple[str, ...]:
    names = [f"{path}.weight", f"{path}.{RECORD}"]
    names += [f"{path}.{s}" for s in (*SCALE_KEYS, *IGNORED)]
    if fmt == "w4a8":
        names += [f"{path}.weight_{p}" for p in W4A8_PARTS]
    return tuple(n for n in names if n in keys)


def comfy_layers(
    keys: list[str],
    get: Callable[[str], Tensor],
    dtype_of: Callable[[str], torch.dtype | None],
) -> dict[str, ComfyLayer]:
    """{module path: ComfyLayer} of every quantized layer among `keys` (the records are
    read through `get`, tensors through nothing else; `dtype_of` reads a header dtype);
    ValueError naming the layer for a record of an unknown format."""
    have = set(keys)
    out = {}
    for key in keys:
        if not key.endswith(f".{RECORD}"):
            continue
        path = key[: -len(RECORD) - 1]
        record = read_record(get(key))
        name = record.get("format")
        if name not in FORMATS:
            raise ValueError(
                f"{key}: ComfyUI quant format {name!r} not in {tuple(FORMATS)} "
                f"(record {json.dumps(record)})"
            )
        fmt = FORMATS[name]
        rotation = (
            int(record.get("convrot_groupsize", 0)) if record.get("convrot") else 0
        )
        group = int(record.get("group_size", 0)) if fmt == "w4a8" else 0
        out[path] = ComfyLayer(path, fmt, rotation, group, _owned(path, have, fmt))
    for key in keys:
        if not key.endswith(".weight") or dtype_of(key) not in CODE_DTYPES:
            continue
        path = key[: -len(".weight")]
        if path in out or not any(f"{path}.{s}" in have for s in SCALE_KEYS):
            continue
        fmt = CODE_DTYPES[dtype_of(key)]
        out[path] = ComfyLayer(path, fmt, 0, 0, _owned(path, have, fmt))
    return out


def _scale(layer: ComfyLayer, get, rows: int) -> Tensor:
    found = [
        f"{layer.path}.{s}" for s in SCALE_KEYS if f"{layer.path}.{s}" in layer.keys
    ]
    if not found:
        raise KeyError(f"{layer.path}.weight is quantized but has no scale tensor")
    scale = get(found[0]).float().reshape(-1)
    return scale.expand(rows).clone() if scale.numel() == 1 else scale


def layer_base(layer: ComfyLayer, get: Callable[[str], Tensor]):
    """The base of `layer` read through `get`: QuantWeight (int8 / fp8) or
    W4A8Weight."""
    q = get(f"{layer.path}.weight")
    if layer.fmt != "w4a8":
        return QuantWeight(q, _scale(layer, get, q.shape[0]), layer.fmt, layer.rotation)
    parts = {}
    for p in W4A8_PARTS:
        key = f"{layer.path}.weight_{p}"
        parts[p] = get(key) if key in layer.keys else None
    return W4A8Weight(
        q,
        parts["s_rel"],
        parts["s_channel"],
        parts["codebook"],
        layer.group,
        layer.rotation,
        parts["correction"],
    )


def stored_format(layers: dict[str, ComfyLayer]) -> str | None:
    """The one format of `layers` (None: no layer; `mixed` for several)."""
    formats = {layer.fmt for layer in layers.values()}
    if not formats:
        return None
    return formats.pop() if len(formats) == 1 else "mixed"


@torch.no_grad()
def attach_layers(
    model: nn.Module,
    layers: dict[str, ComfyLayer],
    get: Callable[[str], Tensor],
    device,
    dtype: torch.dtype,
) -> int:
    """Attach the base of every layer (read through `get`, moved to `device`, bound for
    `dtype`) to its MossLinear; returns how many."""
    for path, layer in layers.items():
        lin = model.get_submodule(path)
        base = layer_base(layer, get).to(device)
        lin.weight = nn.Parameter(
            torch.empty(0, dtype=dtype, device=device), requires_grad=False
        )
        lin.set_base(base)
        base.bind(dtype, device)
    return len(layers)
