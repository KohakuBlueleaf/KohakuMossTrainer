"""Key normalisation between checkpoint formats and the model's own names."""

import re

import torch
from torch import Tensor

PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")
DERIVED = ("rope.bands",)
FP16_FIX_KEY = "kohakumoss_fp16_fix"
FUSED_QKV = re.compile(r"^(layers\.\d+\.proj)\.qkv\.weight$")
FUSED_QKV_PARTS = re.compile(r"^(layers\.\d+\.proj)\.qkv\.(weight|weight_scale|comfy_quant)$")  # fmt: skip
UNSPLIT = ("comfy_quant",)
RUNTIME_HINTS = re.compile(r"^layers\.\d+\.comfy_attention\.")


def strip_prefix(keys: list[str]) -> str:
    """The format prefix every key carries ('' if none)."""
    for prefix in PREFIXES:
        if keys and all(k.startswith(prefix) for k in keys):
            return prefix
    return ""


def split_fused_qkv(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """`layers.N.proj.qkv.weight` [3D, D] -> q / k / v weights (q;k;v row order)."""
    out = {}
    for key, value in sd.items():
        m = FUSED_QKV.match(key)
        if m is None:
            out[key] = value
            continue
        q, k, v = value.chunk(3, 0)
        out[f"{m.group(1)}.q.weight"] = q
        out[f"{m.group(1)}.k.weight"] = k
        out[f"{m.group(1)}.v.weight"] = v
    return out


def fuse_qkv(sd: dict[str, Tensor], depth: int) -> dict[str, Tensor]:
    """Inverse of `split_fused_qkv` for the ComfyUI fused layout."""
    out = dict(sd)
    for i in range(depth):
        base = f"layers.{i}.proj"
        parts = [out.pop(f"{base}.{n}.weight") for n in ("q", "k", "v")]
        out[f"{base}.qkv.weight"] = torch.cat(parts, 0)
    return out
