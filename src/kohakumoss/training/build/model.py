"""Model-side build: backbone, fold, memory plan, trainable set, encoders, export."""

import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

from kohakumoss.adapters.apply import apply as apply_adapters
from kohakumoss.adapters.spec import spec_from_knobs
from kohakumoss.encoders.text import QwenTextEncoder
from kohakumoss.encoders.vae import FluxVae
from kohakumoss.model import build_backbone as build_moss
from kohakumoss.model import load_backbone, resolve_config
from kohakumoss.model.numerics import output_row_scales
from kohakumoss.model.numerics.fp16fix import apply_fp16_fix
from kohakumoss.model.opset import OpSet
from kohakumoss.training.build.smoke import RandomLatentEncoder, RandomTextEncoder
from kohakumoss.training.full import FullTrainable
from kohakumoss.training.knobs import memory_overrides, plan_problems, run_name
from kohakumoss.training.memory.apply import apply_to_backbone
from kohakumoss.training.memory.estimate import Workload
from kohakumoss.training.memory.planner import auto_plan
from kohakumoss.training.memory.profiles import resolve as resolve_profile
from kohakumoss.training.optim.groups import last_block_qk_norm
from kohakumoss.training.precision import apply_param_dtypes
from kohakumoss.types.plans import PrecisionPolicy


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
    cfg = resolve_config(k["VARIANT"], k["ARCH_OVERRIDES"] or None)
    if k["WEIGHTS"] == "random":
        return build_moss(
            cfg,
            device=device,
            dtype=policy.compute,
            keep_fp32=policy.keep_fp32,
            kernels=k["KERNELS"],
            init=True,
            seed=k["SEED"],
        )
    model, report = load_backbone(
        k["WEIGHTS"],
        device=device,
        dtype=policy.compute,
        keep_fp32=policy.keep_fp32,
        cfg=cfg,
        kernels=k["KERNELS"],
    )
    return model


TEXT_TOKENS_PER_SAMPLE = 300


def workload(k: dict, compute: torch.dtype, world: int = 1):
    """The `auto` planner's per-rank micro-step: MOSS arch of VARIANT, tokens from the
    batching knobs (text length per sample: TEXT_TOKENS_PER_SAMPLE)."""
    cfg = resolve_config(k["VARIANT"], k["ARCH_OVERRIDES"] or None)
    per_img = max(1, k["TARGET_AREA"] // (16 * 16))
    if k["BATCHING"] == "count":
        samples = k["BATCH_SIZE"]
        img, txt = samples * per_img, samples * TEXT_TOKENS_PER_SAMPLE
    else:
        share = per_img / (per_img + TEXT_TOKENS_PER_SAMPLE)
        img = int(k["MAX_TOKENS"] * share)
        txt = k["MAX_TOKENS"] - img
    return Workload(
        arch=cfg,
        img_tokens=img,
        txt_tokens=txt,
        full_ft=k["ALGO"] == "full",
        compute_bytes=torch.empty((), dtype=compute).element_size(),
        world=world,
        sharded=k["PARALLEL"] == "fsdp",
    )


def resolve_memory(k: dict, device: torch.device, compute=torch.bfloat16, world=1):
    """(MemoryPlan, `auto` planner rows or []) for the knobs on `device`."""
    trainable_base = k["ALGO"] == "full"
    overrides = memory_overrides(k)
    if k["MEMORY_PROFILE"] == "auto":
        plan, report = auto_plan(
            workload(k, compute, world),
            device,
            overrides,
            trainable_base=trainable_base,
            reject=partial(plan_problems, k),
        )
        print(f"MEMORY_PROFILE='auto' -> {report.chosen}", file=sys.stderr)
        return plan, report.rows
    plan = resolve_profile(
        k["MEMORY_PROFILE"], overrides, device=device, trainable_base=trainable_base
    )
    return plan, []


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
    return apply_adapters(
        backbone, spec_from_knobs(k), dtype=policy.master, device=device
    )


def build_model(
    k: dict, device: torch.device, policy: PrecisionPolicy, world: int = 1
) -> Built:
    """Build order of types/api.py up to (not including) the parallel wrap and compile:
    load on the host -> FP16_FIX -> memory plan (quantize, place or stream onto
    `device`, offload, runner) -> trainable -> parameter dtypes -> bind_ops -> encoders.
    """
    backbone = build_backbone(k, torch.device("cpu"), policy)
    apply_fp16_fix(backbone, policy.fp16_fix, **policy.fp16_fix_kwargs)
    plan, auto_rows = resolve_memory(k, device, policy.compute, world)
    full = is_direct_full(k)
    engines = apply_to_backbone(
        backbone,
        plan,
        device,
        policy.compute,
        training=True,
        trainable_base=full,
        keep_fp32=policy.keep_fp32,
    )
    if auto_rows:
        engines.reports["auto"] = auto_rows
    trainable = build_trainable(k, backbone, policy, device)
    apply_param_dtypes(backbone, policy, _ids(trainable), k["OPTIMIZER"] == "adamw16")
    engines.exclude_params(trainable.trainable_params())
    backbone.bind_ops(OpSet.resolve(device, policy.compute, k["KERNELS"]))
    vae, text = build_encoders(k, device, policy)
    return Built(backbone, trainable, plan, engines, vae, text)


def _component_path(k: dict, knob: str, sub: str) -> str:
    return k[knob] or (
        os.path.join(k["WEIGHTS"], sub) if k["WEIGHTS"] != "random" else ""
    )


def build_encoders(k: dict, device: torch.device, policy: PrecisionPolicy):
    """VAE / text encoder when latents / text are encoded online or samples are drawn."""
    need_vae = k["LATENT_CACHE"] != "offline" or k["SAMPLE_EVERY"] > 0
    need_text = k["TEXT_CACHE"] != "offline" or k["SAMPLE_EVERY"] > 0
    if k["WEIGHTS"] == "random":
        cfg = resolve_config(k["VARIANT"], k["ARCH_OVERRIDES"] or None)
        vae = RandomLatentEncoder(cfg.latent_channels, seed=k["SEED"], device=device)
        return vae, RandomTextEncoder(cfg.text_dim, device=device)
    vae = text = None
    vae_path = _component_path(k, "VAE", "vae")
    text_path = _component_path(k, "TEXT_ENCODER", "text_encoder")
    if need_vae and vae_path:
        vae = FluxVae.from_pretrained(vae_path, device=device, mode=k["LATENT_DRAW"])
    if need_text and text_path:
        text = QwenTextEncoder.from_pretrained(
            text_path, device=k["TEXT_DEVICE"] or device, output_device=device
        )
    return vae, text


def export(k: dict, built: Built, output_dir: str, metadata: dict) -> list[Path]:
    """Every EXPORT_FORMATS output under `<output>/export/`."""
    root = Path(output_dir) / "export"
    root.mkdir(parents=True, exist_ok=True)
    name = run_name(k)
    out = []
    order = ("lycoris", "comfy", "merged", "diffusers")
    for fmt in sorted(k["EXPORT_FORMATS"], key=order.index):
        suffix = "" if fmt == "diffusers" else ".safetensors"
        path = root / f"{name}-{fmt}{suffix}"
        built.trainable.export(
            str(path), fmt, metadata, row_scales=output_row_scales(built.backbone)
        )
        out.append(path)
    return out
