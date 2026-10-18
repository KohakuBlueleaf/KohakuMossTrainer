"""Save the transformer: one bare-key safetensors (split or fused qkv) or diffusers shards."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor

from kohakumoss.model.backbone import MossBackbone
from kohakumoss.model.io.keymap import FP16_FIX_KEY, fuse_qkv
from kohakumoss.model.io.load import INDEX, base_keys
from kohakumoss.model.linear import MossLinear
from kohakumoss.model.numerics.fold import fold_rows, unscaled
from kohakumoss.types.arch import MossArchConfig


def fp16_fix_metadata(model: MossBackbone, extra: dict | None = None) -> dict[str, str]:
    """Safetensors metadata recording a weight fold the saved tensors carry."""
    if model.fold_scales is None:
        return {}
    info = {
        "c": model.fold_scales[0],
        "via": model.fold_via,
        "s_attn": model.fold_pregate[0],
        "s_ffn": model.fold_pregate[1],
        "eps": "stock",
        **(extra or {}),
    }
    return {FP16_FIX_KEY: json.dumps(info)}


def export_state(
    model: MossBackbone,
    dtype: torch.dtype | None = torch.bfloat16,
    fused_qkv: bool = False,
    prefix: str = "",
    keep_fold: bool = False,
    weights: dict[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Base weights only, cast (rope.bands stays fp32), contiguous on CPU, read one
    tensor at a time from the live parameters and buffers (full under
    `parallel.fsdp.unsharded`). Stored (quantized / streamed) bases are materialised
    dense one layer at a time; `weights` (state keys -> tensors in the model's own
    folded space, e.g. merged W + delta W) replace the model's entries; the fold is
    undone unless `keep_fold` (the fp16-fix checkpoint, which holds a uniform gate fold
    only: a per-block fold raises)."""
    if keep_fold and model.fold_scales is not None and model.fold_via != "gate":
        raise ValueError(
            f"keep_fold writes a uniform gate fold; this model is folded via "
            f"{model.fold_via!r} (per-block factors exist only as an online fix)"
        )
    live = {**dict(model.named_buffers()), **dict(model.named_parameters())}
    stored = {
        f"{path}.weight": module.base
        for path, module in model.named_modules()
        if isinstance(module, MossLinear) and module.base is not None
    }
    weights = weights or {}
    rows = {} if keep_fold else fold_rows(model)
    out = {}
    for k in base_keys(model.cfg):
        if k in weights:
            t = weights[k]
        elif k in stored:
            t = stored[k].materialize()
        else:
            t = live[k]
        t = t.detach()
        if k in rows:
            t = unscaled(t, rows[k])
        if dtype is not None and k != "rope.bands" and t.is_floating_point():
            t = t.to(dtype)
        out[k] = t.contiguous().cpu()
    if fused_qkv:
        out = fuse_qkv(out, model.cfg.depth)
    return {prefix + k: v for k, v in out.items()}


def save_single(
    model: MossBackbone,
    path: str | Path,
    dtype: torch.dtype | None = torch.bfloat16,
    fused_qkv: bool = False,
    prefix: str = "",
    metadata: dict[str, str] | None = None,
    keep_fold: bool = False,
) -> None:
    """One safetensors file (ComfyUI "Load Diffusion Model" accepts bare keys).
    `keep_fold` writes the folded weights and records the fold in the metadata."""
    meta = dict(metadata or {})
    if keep_fold:
        meta.update(fp16_fix_metadata(model))
    state = export_state(model, dtype, fused_qkv, prefix, keep_fold)
    save_file(state, str(path), meta or None)


def save_diffusers(
    model: MossBackbone,
    out_dir: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    shard_bytes: int = 5_000_000_000,
    keep_fold: bool = False,
) -> None:
    """HF transformer folder: config.json + shards + index (official layout).
    `keep_fold` writes the folded weights and records the fold on every shard and in
    the index (the fp16-fix checkpoint as a diffusers dir)."""
    state = export_state(model, dtype, keep_fold=keep_fold)
    meta = fp16_fix_metadata(model) if keep_fold else {}
    write_diffusers(state, out_dir, model.cfg, shard_bytes, meta or None)


def write_diffusers(
    state: dict[str, Tensor],
    out_dir: str | Path,
    cfg: MossArchConfig,
    shard_bytes: int = 5_000_000_000,
    metadata: dict[str, str] | None = None,
) -> None:
    """Shards + index + config.json of `cfg` for a ready state dict (bare keys);
    `metadata` goes on every shard (with format=pt) and into the index metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Tensor]] = [{}]
    size = 0
    for k, t in state.items():
        nbytes = t.nbytes
        if shards[-1] and size + nbytes > shard_bytes:
            shards.append({})
            size = 0
        shards[-1][k] = t
        size += nbytes
    n = len(shards)
    weight_map = {}
    shard_meta = {"format": "pt", **metadata} if metadata else None
    for i, shard in enumerate(shards, 1):
        name = f"diffusion_pytorch_model-{i:05d}-of-{n:05d}.safetensors"
        save_file(shard, str(out_dir / name), shard_meta)
        weight_map.update({k: name for k in shard})
    total = sum(t.nbytes for t in state.values())
    meta = {"total_size": total, **(metadata or {})}
    index = {"metadata": meta, "weight_map": weight_map}
    (out_dir / INDEX).write_text(json.dumps(index, indent=2), encoding="utf-8")
    config = {
        "_class_name": "MOSSImageTransformer",
        "_diffusers_version": "0.38.0",
        "dim": cfg.dim,
        "ffn_dim": cfg.ffn_dim,
        "freq_dim": cfg.freq_dim,
        "latent_channels": cfg.latent_channels,
        "num_heads": cfg.heads,
        "num_layers": cfg.depth,
        "rope_base": cfg.rope_base,
        "text_dim": cfg.text_dim,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
