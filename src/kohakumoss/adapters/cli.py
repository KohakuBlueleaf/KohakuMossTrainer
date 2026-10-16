"""Entry points of scripts/export: convert a training state, or merge adapter files."""

from dataclasses import fields

import torch

from kohakumoss.adapters.adapter_set import RAW_TAG
from kohakumoss.adapters.apply import apply
from kohakumoss.adapters.formats import read, spec_from_meta
from kohakumoss.adapters.inference import load_adapters
from kohakumoss.adapters.spec import AdapterSpec
from kohakumoss.model import load_backbone, save_diffusers, save_single
from kohakumoss.model.numerics import apply_fp16_fix, output_row_scales

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _problems(cfg: dict, required: tuple[str, ...]) -> None:
    missing = [k for k in required if not cfg.get(k)]
    if cfg.get("DTYPE", "bf16") not in DTYPES:
        missing.append(f"DTYPE={cfg['DTYPE']!r} not in {sorted(DTYPES)}")
    if missing:
        raise SystemExit("export: " + "; ".join(f"{k} is required" for k in missing))


def _spec(meta: dict, multiplier: float) -> AdapterSpec:
    known = {f.name for f in fields(AdapterSpec)}
    raw = {k: v for k, v in (spec_from_meta(meta) or {}).items() if k in known}
    quiet = {"dropout": 0.0, "rank_dropout": 0.0, "module_dropout": 0.0}
    return AdapterSpec(**{**raw, **quiet, "multiplier": multiplier})


def export_main(cfg: dict) -> None:
    """ADAPTER (a raw training state) on WEIGHTS -> OUT in FORMAT."""
    _problems(cfg, ("WEIGHTS", "ADAPTER", "OUT"))
    dtype = DTYPES[cfg.get("DTYPE", "bf16")]
    tensors, meta = read(cfg["ADAPTER"])
    if not meta.get(RAW_TAG):
        raise SystemExit("export: ADAPTER must be a training state (save_state output)")
    model, _ = load_backbone(cfg["WEIGHTS"], device=cfg.get("DEVICE", "cpu"))
    model.requires_grad_(False)
    if cfg.get("FP16_FIX", "none") == "weights":
        apply_fp16_fix(model, "weights", **cfg.get("FP16_FIX_KWARGS", {}))
    aset = apply(model, _spec(meta, float(cfg.get("MULTIPLIER", 1.0))))
    aset.load_state_dict(tensors)
    fmt = cfg.get("FORMAT", "comfy")
    scales = output_row_scales(model)
    match fmt:
        case "comfy" | "lycoris":
            aset.export(
                cfg["OUT"],
                fmt,
                style=cfg.get("STYLE", "lora_unet"),
                dtype=dtype,
                row_scales=scales,
            )
        case "merged":
            aset.merge_into(aset.multiplier)
            save_single(model, cfg["OUT"], dtype, bool(cfg.get("FUSED_QKV", False)))
        case "diffusers":
            aset.merge_into(aset.multiplier)
            save_diffusers(model, cfg["OUT"], dtype)
        case _:
            raise SystemExit(f"export: FORMAT={fmt!r} not supported")


def merge_main(cfg: dict) -> None:
    """ADAPTERS [(file, multiplier), ...] of any format merged into WEIGHTS -> OUT."""
    _problems(cfg, ("WEIGHTS", "ADAPTERS", "OUT"))
    dtype = DTYPES[cfg.get("DTYPE", "bf16")]
    model, _ = load_backbone(cfg["WEIGHTS"], device=cfg.get("DEVICE", "cpu"))
    model.requires_grad_(False)
    adapters = [(path, float(m)) for path, m in cfg["ADAPTERS"]]
    load_adapters(model, adapters, mode="merge")
    if cfg.get("FORMAT", "merged") == "diffusers":
        save_diffusers(model, cfg["OUT"], dtype)
    else:
        save_single(model, cfg["OUT"], dtype, bool(cfg.get("FUSED_QKV", False)))
