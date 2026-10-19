"""The stored format of a ComfyUI Qwen3.5 text-encoder file (`comfy_files`) from its
headers and quant records; the streamed load (`encoders.text.load`) reads its
tensors. See docs/guides/weights.md#text-encoder."""

from pathlib import Path

import torch
from safetensors import safe_open

from kohakumoss.encoders.text.comfy_files import comfy_text_key
from kohakumoss.quant.comfy_format import ComfyLayer, comfy_layers
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
