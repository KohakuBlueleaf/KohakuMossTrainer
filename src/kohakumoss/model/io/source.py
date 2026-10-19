"""Checkpoint tensor sources: headers first, tensors read one at a time, model-key view."""

import json
from pathlib import Path
from typing import Protocol

import torch
from safetensors import safe_open
from torch import Tensor

from kohakumoss.model.io.keymap import (
    FUSED_QKV_PARTS,
    RUNTIME_HINTS,
    UNSPLIT,
    strip_prefix,
)
from kohakumoss_kernels.dtypes import SAFETENSORS as SAFETENSORS_DTYPES

INDEX = "diffusion_pytorch_model.safetensors.index.json"


class TensorSource(Protocol):
    def keys(self) -> list[str]: ...

    def shape(self, key: str) -> tuple[int, ...]: ...

    def get(self, key: str) -> Tensor: ...


class DictSource:
    """An in-memory state dict as a source."""

    def __init__(self, state: dict[str, Tensor]):
        self.state = state

    def keys(self) -> list[str]:
        return list(self.state)

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self.state[key].shape)

    def get(self, key: str) -> Tensor:
        return self.state[key]


def transformer_dir(path: Path) -> Path:
    if (path / "transformer" / INDEX).is_file() or (path / "transformer").is_dir():
        return path / "transformer"
    return path


def checkpoint_files(path: str | Path) -> list[Path]:
    """The safetensors files of a single file, an indexed shard set or a one-file dir."""
    path = Path(path)
    if path.is_file():
        return [path]
    tdir = transformer_dir(path)
    if (tdir / INDEX).is_file():
        index = json.loads((tdir / INDEX).read_text(encoding="utf-8"))
        return [tdir / s for s in sorted(set(index["weight_map"].values()))]
    files = sorted(tdir.glob("*.safetensors"))
    if len(files) != 1:
        raise FileNotFoundError(f"no unique safetensors checkpoint under {tdir}")
    return files


class CheckpointSource:
    """Safetensors files opened lazily: keys and shapes from headers, one tensor per get."""

    def __init__(self, path: str | Path):
        self._where = {}
        for f in checkpoint_files(path):
            handle = safe_open(str(f), framework="pt", device="cpu")
            for key in handle.keys():
                self._where[key] = handle

    def keys(self) -> list[str]:
        return list(self._where)

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._where[key].get_slice(key).get_shape())

    def dtype(self, key: str) -> torch.dtype | None:
        return SAFETENSORS_DTYPES.get(self._where[key].get_slice(key).get_dtype())

    def get(self, key: str) -> Tensor:
        return self._where[key].get_tensor(key)


class ModelKeyView:
    """Model-key view of a source: format prefix stripped, fused `proj.qkv` split into
    q / k / v (its weight and row scales by rows, its ComfyUI quant record shared; the
    fused tensor is read once per layer), ComfyUI runtime hints (`comfy_attention.*`,
    counted in `hints`) left out."""

    def __init__(self, source: TensorSource):
        self.source = source
        raw = source.keys()
        self.prefix = strip_prefix(raw)
        self.n_source = len(raw)
        self.hints = 0
        self.map: dict[str, tuple[str, int | None]] = {}
        for key in raw:
            name = key[len(self.prefix) :]
            if RUNTIME_HINTS.match(name):
                self.hints += 1
                continue
            m = FUSED_QKV_PARTS.match(name)
            if m is None:
                self.map[name] = (key, None)
                continue
            tail = m.group(2)
            for i, part in enumerate("qkv"):
                index = None if tail in UNSPLIT else i
                self.map[f"{m.group(1)}.{part}.{tail}"] = (key, index)
        self._fused: tuple[str, Tensor] | None = None

    def keys(self) -> list[str]:
        return list(self.map)

    def shape(self, key: str) -> tuple[int, ...]:
        src, part = self.map[key]
        s = self.source.shape(src)
        return s if part is None else (s[0] // 3, *s[1:])

    def dtype(self, key: str) -> torch.dtype | None:
        return self.source.dtype(self.map[key][0])

    def get(self, key: str) -> Tensor:
        src, part = self.map[key]
        if part is None:
            return self.source.get(src)
        if self._fused is None or self._fused[0] != src:
            self._fused = (src, self.source.get(src))
        return self._fused[1].chunk(3, 0)[part]
