"""Load MOSS-Image transformer weights from HF shards, a ComfyUI file or our own saves."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor, nn

from kohakumoss.model.backbone import MossBackbone
from kohakumoss.model.build import DEFAULT_KEEP_FP32, build_backbone
from kohakumoss.model.io.keymap import DERIVED, FP16_FIX_KEY
from kohakumoss.model.io.source import (
    INDEX,
    CheckpointSource,
    DictSource,
    ModelKeyView,
    checkpoint_files,
    transformer_dir,
)
from kohakumoss.model.numerics.fold import adopt_fold
from kohakumoss.model.opset import OpSet
from kohakumoss.types.arch import MossArchConfig

__all__ = [
    "INDEX",
    "LoadReport",
    "base_keys",
    "infer_config",
    "load_backbone",
    "load_state_into",
    "read_config",
    "read_fp16_fix",
    "read_state",
]


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


def read_state(
    path: str | Path, device: torch.device | str = "cpu"
) -> dict[str, Tensor]:
    """Every tensor of a .safetensors file or an HF transformer dir, in memory."""
    state = {}
    for f in checkpoint_files(path):
        state.update(load_file(str(f), device=str(device)))
    return state


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
    cfg = transformer_dir(path) / "config.json"
    return json.loads(cfg.read_text(encoding="utf-8")) if cfg.is_file() else None


def _view(state) -> ModelKeyView:
    if isinstance(state, ModelKeyView):
        return state
    return ModelKeyView(DictSource(state) if isinstance(state, dict) else state)


def infer_config(state, hf: dict | None = None) -> MossArchConfig:
    """Architecture from tensor shapes (head_dim from the qk-norm weight); `state` is a
    state dict, a tensor source or a ModelKeyView (headers only are read)."""
    view = _view(state)
    keys = view.keys()
    dim = view.shape("shared_block_modulation.weight")[0] // 6
    heads = dim // view.shape("layers.0.q_norm.weight")[0]
    depth = 1 + max(int(k.split(".")[1]) for k in keys if k.startswith("layers."))
    text_blocks = sum(
        1
        for k in keys
        if k.startswith("text_embedder.blocks.") and k.endswith("qkv.weight")
    )
    return MossArchConfig(
        dim=dim,
        depth=depth,
        heads=heads,
        ffn_dim=view.shape("layers.0.proj.ffn_up_gate.weight")[0] // 2,
        text_dim=view.shape("text_embedder.proj.weight")[1],
        text_blocks=text_blocks,
        latent_channels=view.shape("video_embedder.weight")[1],
        freq_dim=view.shape("time_embedding.0.weight")[1],
        rope_base=float(hf["rope_base"]) if hf else 10000.0,
        negative_skip_layers=(22,) if depth == 42 else (),
    )


def _assign(model: nn.Module, key: str, value: Tensor) -> None:
    path, _, name = key.rpartition(".")
    module = model.get_submodule(path)
    if name in module._parameters:
        old = module._parameters[name]
        module._parameters[name] = nn.Parameter(value, requires_grad=old.requires_grad)
    else:
        module._buffers[name] = value


def load_state_into(
    model: MossBackbone,
    state,
    source: str = "",
    strict: bool = True,
    device: torch.device | str | None = None,
) -> LoadReport:
    """Assign a state dict or tensor source into `model` (meta or materialised), one
    tensor at a time, each cast to its param's dtype and moved to `device` (default:
    where the source tensor is)."""
    view = _view(state)
    names = view.keys()
    expected = base_keys(model.cfg)
    current = model.state_dict()
    report = LoadReport(source, view.prefix, view.n_source)
    have = set(names)
    report.missing = [k for k in expected if k not in have and k not in DERIVED]
    report.derived = [k for k in DERIVED if k not in have]
    report.unexpected = [k for k in names if k not in current]
    if strict and not report.ok:
        raise KeyError(
            f"checkpoint {source}: missing {report.missing[:8]} "
            f"({len(report.missing)}), unexpected {report.unexpected[:8]} "
            f"({len(report.unexpected)})"
        )
    for k in names:
        if k not in current or k in DERIVED:
            continue
        t = view.get(k)
        dst = device if device is not None else t.device
        _assign(model, k, t.to(device=dst, dtype=current[k].dtype))
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
    """Build on meta, then read, cast and move one tensor at a time (host and device
    peaks are one tensor plus the model), bind ops. An fp16-fix file is recorded as
    folded (savers and adapter exports undo it); `fp16_fix_eps="exact"` uses eps * c^2.
    """
    view = ModelKeyView(CheckpointSource(path))
    cfg = cfg or infer_config(view, read_config(path))
    model = build_backbone(cfg, device="meta", dtype=dtype, keep_fp32=keep_fp32)
    report = load_state_into(model, view, str(path), strict, device=device)
    model.rope.to(device)
    fix = read_fp16_fix(path)
    if fix is not None:
        adopt_fold(model, fix["c"], (fix["s_attn"], fix["s_ffn"]), fp16_fix_eps)
        model.fp16_fix = "weights"
    model.bind_ops(OpSet.resolve(device, dtype, kernels))
    return model, report
