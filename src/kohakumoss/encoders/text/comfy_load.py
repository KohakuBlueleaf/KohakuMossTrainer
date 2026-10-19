"""Load a ComfyUI Qwen3.5 text-encoder file (`comfy_files`): the text model built from
the packaged config, its tensors read from `model.language_model.*`, its quantized
layers (int8 / W4A8, ConvRot) attached as QuantLinear / QuantEmbedding, or dequantized
into the dense model when the run's precision is another one. See
docs/guides/weights.md#text-encoder."""

from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoModel
from transformers.initialization import no_init_weights

from kohakumoss.encoders.text.comfy_files import (
    COMFY_TEXT_PREFIX,
    comfy_text_key,
    packaged_config,
)
from kohakumoss.quant.comfy_format import ComfyLayer, comfy_layers, layer_base
from kohakumoss.quant.convert import QuantLinear
from kohakumoss.quant.embedding import QuantEmbedding
from kohakumoss.quant.source import SafetensorsFiles, SourceWeights
from kohakumoss_kernels.dtypes import SAFETENSORS as SAFETENSORS_DTYPES


class _TextFile:
    """A ComfyUI text-encoder file by the text model's own keys (headers first, one
    tensor per `get`)."""

    def __init__(self, handle):
        self.handle = handle
        self.names = {}
        for key in handle.keys():
            name = comfy_text_key(key)
            if name is not None:
                self.names[name] = key

    def keys(self) -> list[str]:
        return list(self.names)

    def get(self, name: str):
        return self.handle.get_tensor(self.names[name])

    def dtype(self, name: str) -> torch.dtype | None:
        raw = self.handle.get_slice(self.names[name]).get_dtype()
        return SAFETENSORS_DTYPES.get(raw)


def _layers(f: _TextFile) -> dict[str, ComfyLayer]:
    return comfy_layers(f.keys(), f.get, f.dtype)


def text_file_format(path: str | Path) -> tuple[str, str]:
    """(precision, layout) a ComfyUI text-encoder file stores: `compute` and "" for a
    16-bit file, else the linears' format (`int8` / `fp8` / `w4a8`) and their stored
    layouts joined by "+" (e.g. `int8-convrot256`), from the headers and quant records
    only."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        layers = _layers(_TextFile(handle))
    linears = {lay.fmt for p, lay in layers.items() if p != "embed_tokens"}
    if not linears:
        return "compute", ""
    precision = linears.pop() if len(linears) == 1 else "mixed"
    layouts = sorted({lay.layout for lay in layers.values()})
    return precision, "+".join(layouts)


def _under(key: str, paths: set[str]) -> bool:
    parts = key.split(".")
    return any(".".join(parts[:i]) in paths for i in range(1, len(parts)))


def _attach(model: nn.Module, path: str, base, bias, dtype: torch.dtype) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    dense = getattr(parent, name)
    if isinstance(dense, nn.Embedding):
        quant = QuantEmbedding(base.q, base.scale, base.rotation, dtype)
    else:
        quant = QuantLinear.from_base(
            dense.in_features, dense.out_features, base, bias, dtype
        )
    setattr(parent, name, quant)


@torch.no_grad()
def load_comfy_text(path: str | Path, dtype: torch.dtype, keep: bool) -> nn.Module:
    """The Qwen3.5 text model of a ComfyUI file in `dtype`: its quantized layers kept
    as stored bases (`keep`) or dequantized into the dense weights (`keep=False`); the
    vision tower, MTP head and LM head are not read."""
    config = packaged_config().text_config
    with no_init_weights():
        model = AutoModel.from_config(config, dtype=dtype)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        f = _TextFile(handle)
        layers = _layers(f)
        owned = {k for layer in layers.values() for k in layer.keys}
        for p, layer in layers.items():
            base = layer_base(layer, f.get)
            bias = f.get(f"{p}.bias") if f"{p}.bias" in f.names else None
            if keep:
                _attach(model, p, base, bias, dtype)
            else:
                module = model.get_submodule(p)
                module.weight = nn.Parameter(base.weight(dtype), requires_grad=False)
        rest = {k: f.get(k) for k in f.keys() if k not in owned}
        if keep:
            rest = {k: v for k, v in rest.items() if not _under(k, set(layers))}
    missing, unexpected = model.load_state_dict(rest, strict=False)
    paths = set(layers)
    done = {f"{p}.weight" for p in layers}
    left = sorted(
        k for k in missing if k not in done and not (keep and _under(k, paths))
    )
    if left or unexpected:
        raise ValueError(
            f"{path} is not the packaged Qwen3.5 text model: missing {left[:4]}, "
            f"unexpected {list(unexpected)[:4]}"
        )
    return model


def comfy_text_source(path: str | Path) -> SourceWeights:
    """The 16-bit stored weights of a ComfyUI text-encoder file by text-model weight
    name (quantized codes are not values)."""
    tensors = SafetensorsFiles([Path(path)])
    keys = set(tensors.keys())

    def resolve(name: str) -> str | None:
        key = COMFY_TEXT_PREFIX + name
        if key not in keys or tensors.dtype(key) not in (torch.bfloat16, torch.float16):
            return None
        return key

    return SourceWeights(tensors, resolve)
