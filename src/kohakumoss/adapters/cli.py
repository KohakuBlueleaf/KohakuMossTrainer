"""Entry points of scripts/export: convert a training state, or merge adapter files."""

import torch

from kohakumoss.adapters.adapter_set import RAW_TAG
from kohakumoss.adapters.apply import apply
from kohakumoss.adapters.formats import read, spec_from_meta
from kohakumoss.adapters.inference import load_adapters
from kohakumoss.adapters.spec import inference_spec
from kohakumoss.model import load_backbone, save_diffusers, save_single
from kohakumoss.model.numerics import apply_fp16_fix, output_row_scales

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
FORMATS = ("comfy", "lycoris", "merged", "diffusers")


def _problems(cfg: dict, required: tuple[str, ...]) -> None:
    missing = [k for k in required if not cfg.get(k)]
    if cfg.get("DTYPE", "bf16") not in DTYPES:
        missing.append(f"DTYPE={cfg['DTYPE']!r} not in {sorted(DTYPES)}")
    if missing:
        raise SystemExit("export: " + "; ".join(f"{k} is required" for k in missing))


def export_main(cfg: dict) -> None:
    """ADAPTER (a raw training state) on WEIGHTS -> OUT in FORMAT."""
    _problems(cfg, ("WEIGHTS", "ADAPTER", "OUT"))
    dtype = DTYPES[cfg.get("DTYPE", "bf16")]
    fmt = cfg.get("FORMAT", "comfy")
    if fmt not in FORMATS:
        raise SystemExit(f"export: FORMAT={fmt!r} not in {FORMATS}")
    tensors, meta = read(cfg["ADAPTER"])
    if not meta.get(RAW_TAG):
        raise SystemExit("export: ADAPTER must be a training state (save_state output)")
    model, _ = load_backbone(cfg["WEIGHTS"], device=cfg.get("DEVICE", "cpu"))
    model.requires_grad_(False)
    if cfg.get("FP16_FIX", "none") == "weights":
        apply_fp16_fix(model, "weights", **cfg.get("FP16_FIX_KWARGS", {}))
    spec = inference_spec(spec_from_meta(meta) or {}, float(cfg.get("MULTIPLIER", 1.0)))
    aset = apply(model, spec)
    aset.load_state_dict(tensors)
    aset.export(
        cfg["OUT"],
        fmt,
        style=cfg.get("STYLE", "lora_unet"),
        dtype=dtype,
        row_scales=output_row_scales(model),
        fused_qkv=bool(cfg.get("FUSED_QKV", False)),
    )


def merge_main(cfg: dict) -> None:
    """ADAPTERS [(file, multiplier), ...] of any format merged into WEIGHTS -> OUT."""
    _problems(cfg, ("WEIGHTS", "ADAPTERS", "OUT"))
    dtype = DTYPES[cfg.get("DTYPE", "bf16")]
    model, _ = load_backbone(cfg["WEIGHTS"], device=cfg.get("DEVICE", "cpu"))
    model.requires_grad_(False)
    adapters = [(path, float(m)) for path, m in cfg["ADAPTERS"]]
    load_adapters(model, adapters, mode="merge", undoable=False)
    if cfg.get("FORMAT", "merged") == "diffusers":
        save_diffusers(model, cfg["OUT"], dtype)
    else:
        save_single(model, cfg["OUT"], dtype, bool(cfg.get("FUSED_QKV", False)))
