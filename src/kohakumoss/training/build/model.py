"""Model-side build: backbone, fold, memory plan, trainable set, encoders, export."""

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from kohakumoss.registry import import_object
from kohakumoss.training.build import integration as I
from kohakumoss.training.build.smoke import RandomLatentEncoder, RandomTextEncoder
from kohakumoss.training.full import FullTrainable
from kohakumoss.training.optim.groups import last_block_qk_norm
from kohakumoss.training.precision import apply_param_dtypes
from kohakumoss.types.plans import PrecisionPolicy


def optional_object(path: str):
    """The object at `path`, or None when its module is not part of this install."""
    try:
        return import_object(path)
    except (ImportError, AttributeError):
        return None


@dataclass
class Built:
    backbone: torch.nn.Module
    trainable: object
    plan: object
    engines: object
    vae: object | None
    text_encoder: object | None


def build_backbone(k: dict, device: torch.device, policy: PrecisionPolicy):
    """Random tiny/preset model for WEIGHTS='random', else the loaded checkpoint."""
    cfg = import_object(I.MODEL_RESOLVE)(k["VARIANT"], k["ARCH_OVERRIDES"] or None)
    if k["WEIGHTS"] == "random":
        return import_object(I.MODEL_BUILD)(
            cfg,
            device=device,
            dtype=policy.compute,
            keep_fp32=policy.keep_fp32,
            kernels=k["KERNELS"],
            init=True,
            seed=k["SEED"],
        )
    model, report = import_object(I.MODEL_LOAD)(
        k["WEIGHTS"],
        device=device,
        dtype=policy.compute,
        keep_fp32=policy.keep_fp32,
        cfg=cfg,
        kernels=k["KERNELS"],
    )
    return model


def resolve_memory(k: dict, device: torch.device):
    trainable_base = k["ALGO"] == "full"
    return import_object(I.MEMORY_RESOLVE)(
        k["MEMORY_PROFILE"], k["MEMORY_OVERRIDES"], device=device, trainable_base=trainable_base
    )


def _ids(trainable) -> set[int]:
    return {id(p) for p in trainable.trainable_params()}


def is_direct_full(k: dict) -> bool:
    return k["ALGO"] == "full" and k.get("FULL_MODE", "direct") == "direct"


def build_trainable(k: dict, backbone, policy: PrecisionPolicy, device):
    """FullTrainable for direct full FT; otherwise the LyCORIS AdapterSet."""
    if is_direct_full(k):
        frozen = () if k["TRAIN_TEXT_REFINER"] else (r"^text_embedder\.",)
        return FullTrainable(backbone, frozen, last_block_qk_norm(backbone.cfg.depth))
    backbone.requires_grad_(False)
    spec = import_object(I.ADAPTERS_SPEC)(k)
    return import_object(I.ADAPTERS_APPLY)(backbone, spec, dtype=policy.master, device=device)


def build_model(k: dict, device: torch.device, policy: PrecisionPolicy) -> Built:
    """Build order of types/api.py up to (not including) the parallel wrap and compile:
    load -> FP16_FIX -> memory plan (quantize / stream / offload / runner) -> trainable
    -> bind_ops -> parameter dtypes -> encoders."""
    backbone = build_backbone(k, device, policy)
    import_object(I.FP16_FIX)(backbone, policy.fp16_fix, **policy.fp16_fix_kwargs)
    plan = resolve_memory(k, device)
    master_16bit = k["OPTIMIZER"] == "adamw16"
    full = is_direct_full(k)
    if full:
        trainable = build_trainable(k, backbone, policy, device)
        apply_param_dtypes(backbone, policy, _ids(trainable), master_16bit)
    engines = import_object(I.MEMORY_APPLY)(
        backbone,
        plan,
        device,
        policy.compute,
        training=True,
        trainable_base=full,
        keep_fp32=policy.keep_fp32,
    )
    if not full:
        trainable = build_trainable(k, backbone, policy, device)
        apply_param_dtypes(backbone, policy, _ids(trainable), master_16bit)
    backbone.bind_ops()
    vae, text = build_encoders(k, device, policy)
    return Built(backbone, trainable, plan, engines, vae, text)


def _component_path(k: dict, knob: str, sub: str) -> str:
    return k[knob] or (os.path.join(k["WEIGHTS"], sub) if k["WEIGHTS"] != "random" else "")


def build_encoders(k: dict, device: torch.device, policy: PrecisionPolicy):
    """VAE / text encoder when latents / text are encoded online or samples are drawn."""
    need_vae = k["LATENT_CACHE"] != "offline" or k["SAMPLE_EVERY"] > 0
    need_text = k["TEXT_CACHE"] != "offline" or k["SAMPLE_EVERY"] > 0
    if k["WEIGHTS"] == "random":
        cfg = import_object(I.MODEL_RESOLVE)(k["VARIANT"], k["ARCH_OVERRIDES"] or None)
        vae = RandomLatentEncoder(cfg.latent_channels, seed=k["SEED"], device=device)
        return vae, RandomTextEncoder(cfg.text_dim, device=device)
    vae = text = None
    vae_path = _component_path(k, "VAE", "vae")
    text_path = _component_path(k, "TEXT_ENCODER", "text_encoder")
    if need_vae and vae_path:
        vae = import_object(I.VAE).from_pretrained(vae_path, device=device)
    if need_text and text_path:
        text = import_object(I.TEXT_ENCODER).from_pretrained(
            text_path, device=k["TEXT_DEVICE"] or device, output_device=device
        )
    return vae, text


def export(k: dict, built: Built, output_dir: str, metadata: dict) -> list[Path]:
    """Every EXPORT_FORMATS output under `<output>/export/`."""
    root = Path(output_dir) / "export"
    root.mkdir(parents=True, exist_ok=True)
    name = k["RUN_NAME"] or "moss"
    out = []
    order = ("lycoris", "comfy", "merged", "diffusers")
    for fmt in sorted(k["EXPORT_FORMATS"], key=order.index):
        match fmt:
            case "diffusers":
                path = root / f"{name}-diffusers"
                if not isinstance(built.trainable, FullTrainable):
                    built.trainable.merge_into(1.0)
                import_object(I.MODEL_SAVE_DIFFUSERS)(built.backbone, path)
            case _:
                path = root / f"{name}-{fmt}.safetensors"
                built.trainable.export(
                    str(path),
                    fmt,
                    metadata,
                    row_scales=import_object(I.MODEL_ROW_SCALES)(built.backbone),
                )
        out.append(path)
    return out
