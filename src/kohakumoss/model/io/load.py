"""Load MOSS-Image transformer weights from HF shards, a ComfyUI file or our own saves."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor

from kohakumoss.model.backbone import MossBackbone
from kohakumoss.model.build import DEFAULT_KEEP_FP32, build_backbone
from kohakumoss.model.io.keymap import (
    DERIVED,
    FP16_FIX_KEY,
    split_fused_qkv,
    strip_prefix,
)
from kohakumoss.model.numerics.fold import adopt_fold
from kohakumoss.model.opset import OpSet
from kohakumoss.types.arch import MossArchConfig

INDEX = "diffusion_pytorch_model.safetensors.index.json"


@dataclass
class LoadReport:
    """Key accounting of one load: every model key filled, every source key consumed."""

    source: str
    prefix: str
    n_source: int
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    derived: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unexpected


def base_keys(cfg: MossArchConfig) -> list[str]:
    """State-dict keys of a plain MossBackbone of `cfg` (adapter keys excluded)."""
    return list(MossBackbone(cfg, device="meta").state_dict().keys())


def _transformer_dir(path: Path) -> Path:
    if (path / "transformer" / INDEX).is_file() or (path / "transformer").is_dir():
        return path / "transformer"
    return path


def read_state(
    path: str | Path, device: torch.device | str = "cpu"
) -> dict[str, Tensor]:
    """Tensors of a .safetensors file or an HF transformer dir (sharded or single)."""
    path = Path(path)
    if path.is_file():
        return load_file(str(path), device=str(device))
    tdir = _transformer_dir(path)
    if (tdir / INDEX).is_file():
        index = json.loads((tdir / INDEX).read_text(encoding="utf-8"))
        state = {}
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(str(tdir / shard), device=str(device)))
        return state
    files = sorted(tdir.glob("*.safetensors"))
    if len(files) != 1:
        raise FileNotFoundError(f"no unique safetensors checkpoint under {tdir}")
    return load_file(str(files[0]), device=str(device))


def read_fp16_fix(path: str | Path) -> dict | None:
    """The fp16-fix record of a single-file checkpoint, if it carries one."""
    path = Path(path)
    if not path.is_file():
        return None
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    return json.loads(meta[FP16_FIX_KEY]) if FP16_FIX_KEY in meta else None


def read_config(path: str | Path) -> dict | None:
    """The HF transformer config.json next to the weights, if any."""
    path = Path(path)
    if path.is_file():
        return None
    cfg = _transformer_dir(path) / "config.json"
    return json.loads(cfg.read_text(encoding="utf-8")) if cfg.is_file() else None


def infer_config(state: dict[str, Tensor], hf: dict | None = None) -> MossArchConfig:
    """Architecture from tensor shapes (head_dim from the qk-norm weight)."""
    sd = split_fused_qkv(_normalise(state)[0])
    dim = sd["shared_block_modulation.weight"].shape[0] // 6
    heads = dim // sd["layers.0.q_norm.weight"].shape[0]
    depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layers."))
    text_blocks = sum(
        1
        for k in sd
        if k.startswith("text_embedder.blocks.") and k.endswith("qkv.weight")
    )
    return MossArchConfig(
        dim=dim,
        depth=depth,
        heads=heads,
        ffn_dim=sd["layers.0.proj.ffn_up_gate.weight"].shape[0] // 2,
        text_dim=sd["text_embedder.proj.weight"].shape[1],
        text_blocks=text_blocks,
        latent_channels=sd["video_embedder.weight"].shape[1],
        freq_dim=sd["time_embedding.0.weight"].shape[1],
        rope_base=float(hf["rope_base"]) if hf else 10000.0,
        negative_skip_layers=(22,) if depth == 42 else (),
    )


def _normalise(state: dict[str, Tensor]) -> tuple[dict[str, Tensor], str]:
    prefix = strip_prefix(list(state))
    return {k[len(prefix) :]: v for k, v in state.items()}, prefix


def load_state_into(
    model: MossBackbone,
    state: dict[str, Tensor],
    source: str = "",
    strict: bool = True,
    device: torch.device | str | None = None,
) -> LoadReport:
    """Assign `state` into `model` (meta or materialised), each tensor cast to its
    param's dtype and moved to `device` (default: where it already is)."""
    sd, prefix = _normalise(state)
    sd = split_fused_qkv(sd)
    expected = base_keys(model.cfg)
    current = model.state_dict()
    report = LoadReport(source, prefix, len(state))
    report.missing = [k for k in expected if k not in sd and k not in DERIVED]
    report.derived = [k for k in DERIVED if k not in sd]
    report.unexpected = [k for k in sd if k not in current]
    if strict and not report.ok:
        raise KeyError(
            f"checkpoint {source}: missing {report.missing[:8]} "
            f"({len(report.missing)}), unexpected {report.unexpected[:8]} "
            f"({len(report.unexpected)})"
        )
    cast = {}
    for k in [k for k in sd if k in current and k not in DERIVED]:
        target = current[k]
        dst = device if device is not None else sd[k].device
        cast[k] = sd.pop(k).to(device=dst, dtype=target.dtype)
    model.load_state_dict(cast, strict=False, assign=True)
    return report


def load_backbone(
    path: str | Path,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    keep_fp32: tuple[str, ...] = DEFAULT_KEEP_FP32,
    cfg: MossArchConfig | None = None,
    kernels: dict[str, str] | None = None,
    strict: bool = True,
    fp16_fix_eps: str = "stock",
) -> tuple[MossBackbone, LoadReport]:
    """Build on meta, read on CPU, cast and move tensor by tensor (peak device memory is
    the model itself), bind ops. An fp16-fix file is recorded as folded (so savers and
    adapter exports can undo it); `fp16_fix_eps="exact"` uses eps * c^2."""
    state = read_state(path, "cpu")
    cfg = cfg or infer_config(state, read_config(path))
    model = build_backbone(cfg, device="meta", dtype=dtype, keep_fp32=keep_fp32)
    report = load_state_into(model, state, str(path), strict, device=device)
    model.rope.to(device)
    fix = read_fp16_fix(path)
    if fix is not None:
        adopt_fold(model, fix["c"], (fix["s_attn"], fix["s_ffn"]), fp16_fix_eps)
        model.fp16_fix = "weights"
    model.bind_ops(OpSet.resolve(device, dtype, kernels))
    return model, report
