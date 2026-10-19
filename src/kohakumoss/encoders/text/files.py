"""Text-encoder checkpoint tensors by the text model's own names, read as zero-copy
views of the memory-mapped safetensors files (headers first, one tensor per `get`): a
model dir's shards, a ComfyUI file or a pre-quantized file. See
docs/internals/text-encoder.md#streamed-load."""

from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor

from kohakumoss.quant.source import SourceWeights
from kohakumoss_kernels.dtypes import SAFETENSORS as SAFETENSORS_DTYPES

HF_TEXT_PREFIXES = ("model.language_model.", "language_model.", "model.")
HF_SKIPPED = ("visual.", "model.visual.", "mtp.", "lm_head.")
VALUES = (torch.bfloat16, torch.float16, torch.float32)


def hf_text_name(key: str) -> str | None:
    """The text model's name of a model-dir checkpoint key (None: the vision tower,
    MTP head or LM head)."""
    if key.startswith(HF_SKIPPED):
        return None
    for prefix in HF_TEXT_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


class TextFiles:
    """Tensors of safetensors `files` by text-model name (`name_of(file key)`, None:
    not the text model's); ValueError when two keys give one name."""

    def __init__(self, files: list[Path], name_of: Callable[[str], str | None]):
        self.where: dict[str, tuple] = {}
        for f in files:
            handle = safe_open(str(f), framework="pt", device="cpu")
            for key in handle.keys():
                name = name_of(key)
                if name is None:
                    continue
                if name in self.where:
                    raise ValueError(
                        f"{f}: keys {self.where[name][1]!r} and {key!r} both name "
                        f"the text model's {name!r}"
                    )
                self.where[name] = (handle, key)

    def keys(self) -> list[str]:
        return list(self.where)

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def get(self, name: str) -> Tensor:
        """The stored tensor `name` (its stored dtype; a view of the file pages)."""
        if name not in self.where:
            raise KeyError(f"the text-encoder checkpoint holds no tensor {name!r}")
        handle, key = self.where[name]
        return handle.get_tensor(key)

    def dtype(self, name: str) -> torch.dtype | None:
        handle, key = self.where[name]
        return SAFETENSORS_DTYPES.get(handle.get_slice(key).get_dtype())

    def source(
        self, prefix: str = "", values: tuple[torch.dtype, ...] = VALUES
    ) -> SourceWeights:
        """The stored weights stored in a `values` dtype under module path `prefix`, by
        weight name relative to it (`quantize_text_linears` names them so)."""

        def resolve(name: str) -> str | None:
            full = prefix + name
            if full not in self.where or self.dtype(full) not in values:
                return None
            return full

        return SourceWeights(self, resolve)


def hf_text_files(model_dir: Path) -> TextFiles:
    """The text model's tensors of a model dir's safetensors shards."""
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{model_dir} holds no .safetensors files")
    return TextFiles(files, hf_text_name)
