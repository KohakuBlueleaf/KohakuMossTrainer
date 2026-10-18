"""Entry points of scripts/export: convert a training state, or merge adapter files."""

from kohakumoss.adapters.build.apply import apply
from kohakumoss.adapters.build.config import inference_config
from kohakumoss.adapters.infer.formats import config_from_meta, read
from kohakumoss.adapters.infer.inference import load_adapters
from kohakumoss.adapters.train.adapter_set import FORMATS, FULL_TAG, fold_of, state_kind
from kohakumoss.model import load_backbone, save_diffusers, save_single
from kohakumoss.model.numerics import carry_fold, output_row_scales
from kohakumoss_kernels.dtypes import BY_NAME as DTYPES

MERGED = ("merged", "diffusers")
RENAMED = {"DTYPE": "renamed to SAVE_DTYPE (the dtype of the written tensors)"}


def _problems(cfg: dict, required: tuple[str, ...]) -> None:
    problems = [f"{k} is required" for k in required if not cfg.get(k)]
    if cfg.get("SAVE_DTYPE", "bf16") not in DTYPES:
        problems.append(f"SAVE_DTYPE={cfg['SAVE_DTYPE']!r} not in {sorted(DTYPES)}")
    if problems:
        raise SystemExit("export: " + "; ".join(problems))


def _training_space(model, meta: dict) -> None:
    """Put `model` in the weight space the state trained in: the fold its metadata
    records (null: unfolded)."""
    carry_fold(model, fold_of(meta)[1])


def _save_model(model, cfg: dict, fmt: str) -> None:
    dtype = DTYPES[cfg.get("SAVE_DTYPE", "bf16")]
    if fmt == "diffusers":
        save_diffusers(model, cfg["OUT"], dtype)
    else:
        save_single(model, cfg["OUT"], dtype, bool(cfg.get("FUSED_QKV", False)))


def export_main(cfg: dict) -> None:
    """ADAPTER (a training state of this version: a run checkpoint's
    trainable.safetensors or a save_state file, its tag / spec / fold in the metadata)
    on WEIGHTS -> OUT in FORMAT; a full-finetune checkpoint exports as a merged /
    diffusers transformer. A file without that metadata is refused (SystemExit)."""
    _problems(cfg, ("WEIGHTS", "ADAPTER", "OUT"))
    dtype = DTYPES[cfg.get("SAVE_DTYPE", "bf16")]
    fmt = cfg.get("FORMAT", "comfy")
    if fmt not in FORMATS:
        raise SystemExit(f"export: FORMAT={fmt!r} not in {FORMATS}")
    tensors, meta = read(cfg["ADAPTER"])
    try:
        full = state_kind(cfg["ADAPTER"], meta) == FULL_TAG
    except ValueError as err:
        raise SystemExit(f"export: {err}") from None
    if full and fmt not in MERGED:
        raise SystemExit(f"export: a full-finetune checkpoint exports as {MERGED}")
    model, _ = load_backbone(cfg["WEIGHTS"], device=cfg.get("DEVICE", "cpu"))
    model.requires_grad_(False)
    _training_space(model, meta)
    if full:
        unknown = model.load_state_dict(tensors, strict=False).unexpected_keys
        if unknown:
            raise SystemExit(f"export: {len(unknown)} unknown keys, e.g. {unknown[:4]}")
        _save_model(model, cfg, fmt)
        return
    multiplier = float(cfg.get("MULTIPLIER", 1.0))
    config = inference_config(config_from_meta(meta) or {}, multiplier)
    aset = apply(model, config)
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
    """ADAPTERS [(file, multiplier), ...] of any format merged into WEIGHTS -> OUT; on a
    folded WEIGHTS file (fp16 fix) each dW carries the fold's row scales."""
    _problems(cfg, ("WEIGHTS", "ADAPTERS", "OUT"))
    model, _ = load_backbone(cfg["WEIGHTS"], device=cfg.get("DEVICE", "cpu"))
    model.requires_grad_(False)
    adapters = [(path, float(m)) for path, m in cfg["ADAPTERS"]]
    scales = output_row_scales(model)
    load_adapters(model, adapters, mode="merge", row_scales=scales, undoable=False)
    _save_model(model, cfg, cfg.get("FORMAT", "merged"))
