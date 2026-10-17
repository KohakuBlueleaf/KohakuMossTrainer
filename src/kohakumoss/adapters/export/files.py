"""Adapter files: safetensors writing and key styles."""

import os

import torch
from safetensors.torch import save_file
from torch import Tensor

KEY_STYLES = ("lora_unet", "diffusion_model")


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
