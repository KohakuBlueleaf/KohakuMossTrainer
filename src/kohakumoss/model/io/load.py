"""Load MOSS-Image transformer weights from HF shards, a ComfyUI file or our own saves."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from kohakumoss.logging.log import get_logger, hl
from kohakumoss.model.backbone import MossBackbone
from kohakumoss.model.build import DEFAULT_KEEP_FP32, build_backbone
from kohakumoss.model.io.keymap import DERIVED, FP16_FIX_KEY, FUSED_QKV, strip_prefix
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
from kohakumoss.quant.comfy_format import (
    ComfyLayer,
    attach_layers,
    comfy_layers,
    stored_format,
)
from kohakumoss.quant.stored import attach_stored, read_record, stored_keys
from kohakumoss.types.arch import MossArchConfig
from kohakumoss_common.io import read_safetensors_metadata
from kohakumoss_common.units import gib
from kohakumoss_kernels.dtypes import short

log = get_logger(__name__)

__all__ = [
    "INDEX",
    "LoadReport",
    "base_keys",
    "file_layers",
    "infer_config",
    "load_backbone",
    "load_state_into",
    "read_config",
    "read_fp16_fix",
    "stored_precision",
    "stores_fused_qkv",
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


def read_fp16_fix(path: str | Path) -> dict | None:
    """The fp16-fix record a checkpoint carries (a single file, or a diffusers
    transformer dir: every shard, and the index when it has one, must hold the same
    record), else None; a dir whose shards disagree raises ValueError."""
    path = Path(path)
    if path.is_file():
        meta = read_safetensors_metadata(path)
        return json.loads(meta[FP16_FIX_KEY]) if FP16_FIX_KEY in meta else None
    if not path.is_dir():
        return None
    try:
        files = checkpoint_files(path)
    except FileNotFoundError:
        return None
    found = {f.name: read_safetensors_metadata(f).get(FP16_FIX_KEY) for f in files}
    index = transformer_dir(path) / INDEX
    if index.is_file():
        meta = json.loads(index.read_text(encoding="utf-8")).get("metadata", {})
        if FP16_FIX_KEY in meta:
            found[INDEX] = meta[FP16_FIX_KEY]
    records = set(found.values())
    if records == {None}:
        return None
    if len(records) != 1:
        raise ValueError(f"fp16-fix record differs across the files of {path}: {found}")
    return json.loads(records.pop())


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
    stored: frozenset[str] = frozenset(),
    extra: frozenset[str] = frozenset(),
) -> LoadReport:
    """Assign a state dict or tensor source into `model` (meta or materialised), one
    tensor at a time, each cast to its param's dtype and moved to `device` (default:
    where the source tensor is). `stored` keys (quantized weights) and `extra` keys
    (their scales) are accounted for and left to the caller."""
    view = _view(state)
    names = view.keys()
    expected = base_keys(model.cfg)
    current = model.state_dict()
    report = LoadReport(source, view.prefix, view.n_source)
    have = set(names)
    report.missing = [k for k in expected if k not in have and k not in DERIVED]
    report.derived = [k for k in DERIVED if k not in have]
    report.unexpected = [k for k in names if k not in current and k not in extra]
    if strict and not report.ok:
        raise KeyError(
            f"checkpoint {source}: missing {report.missing[:8]} "
            f"({len(report.missing)}), unexpected {report.unexpected[:8]} "
            f"({len(report.unexpected)})"
        )
    for k in names:
        if k not in current or k in DERIVED or k in stored:
            continue
        t = view.get(k)
        dst = device if device is not None else t.device
        _assign(model, k, t.to(device=dst, dtype=current[k].dtype))
    return report


def file_layers(view: ModelKeyView, record: dict | None) -> dict[str, ComfyLayer]:
    """The ComfyUI quantized layers of a checkpoint view (none in a file whose layers
    the KohakuMoss fp8 `record` lists)."""
    if record:
        return {}
    return comfy_layers(view.keys(), view.get, view.dtype)


def stored_precision(path: str | Path) -> str:
    """The base precision a DiT checkpoint stores, from its headers and quant records:
    `compute` (16-bit), the fp8-compute precision of a KohakuMoss fp8 file, or the
    ComfyUI format (`int8`, `fp8`, `w4a8`; `mixed` for several)."""
    record = read_record(path)
    if record:
        return "mxfp8" if record["recipe"] == "mx" else "fp8_compute"
    view = ModelKeyView(CheckpointSource(path))
    return stored_format(file_layers(view, None)) or "compute"


def stores_fused_qkv(path: str | Path) -> bool:
    """The DiT checkpoint stores joint-block q / k / v fused as `layers.N.proj.qkv`
    (the ComfyUI layout), from its headers."""
    source = CheckpointSource(path)
    prefix = strip_prefix(source.keys())
    return any(FUSED_QKV.match(k[len(prefix) :]) for k in source.keys())


def _layer_keys(layers: dict[str, ComfyLayer]) -> tuple[set[str], set[str]]:
    weights = {f"{p}.weight" for p in layers}
    rest = {k for layer in layers.values() for k in layer.keys} - weights
    return weights, rest


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
    A pre-quantized file (KohakuMoss fp8, ComfyUI int8 / fp8 / W4A8, ConvRot included)
    attaches its stored bases (no 16-bit copy is built); ComfyUI runtime hints are
    skipped.
    """
    start = time.perf_counter()
    log.info("loading transformer", path=path, dtype=short(dtype), device=device)
    view = ModelKeyView(CheckpointSource(path))
    record = read_record(path)
    layers = file_layers(view, record)
    cfg = cfg or infer_config(view, read_config(path))
    model = build_backbone(cfg, device="meta", dtype=dtype, keep_fp32=keep_fp32)
    stored, extra = stored_keys(record) if record else (set(), set())
    weights, rest = _layer_keys(layers)
    stored, extra = stored | weights, extra | rest
    report = load_state_into(
        model, view, str(path), strict, device, frozenset(stored), frozenset(extra)
    )
    if record:
        attach_stored(model, view, record, device, dtype)
        log.info(
            f"{hl(len(stored))} fp8 bases ({hl(record['recipe'])}) from "
            f"{hl(path, 'path')}"
        )
    if layers:
        attach_layers(model, layers, view.get, device, dtype)
        kinds = sorted({layer.layout for layer in layers.values()})
        log.info(
            f"{hl(len(layers))} stored bases ({hl(', '.join(kinds))}) from "
            f"{hl(path, 'path')}"
        )
    if view.hints:
        log.info(f"{hl(view.hints)} ComfyUI runtime hint tensors not used")
    model.rope.to(device)
    model.weight_path = str(path)
    fix = read_fp16_fix(path)
    if fix is not None:
        adopt_fold(model, fix["c"], (fix["s_attn"], fix["s_ffn"]), fp16_fix_eps)
        model.fp16_fix = "weights"
    model.bind_ops(OpSet.resolve(device, dtype, kernels))
    size = gib(sum(p.nbytes for p in model.parameters()))
    fold = {"fp16_fold_c": fix["c"]} if fix is not None else {}
    log.info(
        f"transformer loaded from {hl(path, 'path')}",
        tensors=report.n_source,
        gib=round(size, 2),
        secs=round(time.perf_counter() - start, 2),
        **fold,
    )
    return model, report
