"""Adapter and checkpoint files: safetensors writing, key styles, diffusers sharding."""

import json
import os

import torch
from safetensors.torch import save_file
from torch import Tensor

KEY_STYLES = ("lora_unet", "diffusion_model")
SHARD_NAME = "diffusion_pytorch_model-{:05d}-of-{:05d}.safetensors"
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"


def layer_key(path: str, style: str, prefix: str = "lora_unet") -> str:
    """Key stem of a layer: `lora_unet_layers_0_proj_q` or `diffusion_model.layers.0.proj.q`."""
    match style:
        case "lora_unet":
            return f"{prefix}_{path}".replace(".", "_")
        case "diffusion_model":
            return f"diffusion_model.{path}"
        case _:
            raise KeyError(f"key style {style!r} not in {KEY_STYLES}")


def to_storage(
    tensors: dict[str, Tensor], dtype: torch.dtype | None
) -> dict[str, Tensor]:
    """Contiguous CPU copies; floating tensors cast to `dtype` (scalars stay fp32)."""
    out = {}
    for key, value in tensors.items():
        t = value.detach().to("cpu")
        if dtype is not None and t.is_floating_point() and t.dim() > 0:
            t = t.to(dtype)
        out[key] = t.contiguous()
    return out


def write(path: str, tensors: dict[str, Tensor], metadata: dict[str, str]) -> None:
    """One safetensors file; metadata values are stringified."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(tensors, path, metadata={k: str(v) for k, v in metadata.items()})


def write_sharded(
    directory: str,
    tensors: dict[str, Tensor],
    config: dict | None,
    max_shard_bytes: int = 5 * 1024**3,
) -> list[str]:
    """Diffusers-style shards + index (+ config.json when given); returns shard paths."""
    os.makedirs(directory, exist_ok=True)
    shards: list[dict[str, Tensor]] = [{}]
    size = 0
    for key in sorted(tensors):
        nbytes = tensors[key].numel() * tensors[key].element_size()
        if shards[-1] and size + nbytes > max_shard_bytes:
            shards.append({})
            size = 0
        shards[-1][key] = tensors[key]
        size += nbytes
    names, weight_map = [], {}
    for i, shard in enumerate(shards, 1):
        name = SHARD_NAME.format(i, len(shards))
        save_file(shard, os.path.join(directory, name), metadata={"format": "pt"})
        names.append(name)
        weight_map.update({key: name for key in shard})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    with open(os.path.join(directory, INDEX_NAME), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    if config is not None:
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    return [os.path.join(directory, n) for n in names]
