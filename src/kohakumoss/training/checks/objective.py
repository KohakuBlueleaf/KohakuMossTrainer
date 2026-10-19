"""Knob checks of the objective, precision fix, single-file weights and the schedule."""

import os

from kohakumoss.model.io.load import read_fp16_fix
from kohakumoss.training.weights import component
from kohakumoss.weights import DIT

REAL_VARIANTS = ("base", "sft")


def carries_fp16_fix(weights: str) -> bool:
    """True when WEIGHTS is a single file with an fp16-fix fold in its metadata."""
    return bool(weights) and read_fp16_fix(weights) is not None


def dit_file(k: dict) -> str:
    """The run's DiT weights path ("" when a ComfyUI folder holds none)."""
    if not k["WEIGHTS"]:
        return ""
    try:
        return str(component(k, DIT))
    except FileNotFoundError:
        return ""


def objective_problems(k: dict) -> list[str]:
    """LOSS, FP16_FIX against PRECISION / VARIANT / WEIGHTS, and the components a
    single-file WEIGHTS needs."""
    p = []
    if k["LOSS"] == "masked_mse":
        p.append(
            "LOSS='masked_mse' needs a per-token loss mask, and no dataset source "
            "produces one: use LOSS='mse' | 'huber' | 'l1'"
        )
    if (
        k["PRECISION"] == "fp16"
        and k["VARIANT"] in REAL_VARIANTS
        and k["FP16_FIX"] == "none"
        and not carries_fp16_fix(dit_file(k))
    ):
        p.append(
            f"PRECISION='fp16' on VARIANT={k['VARIANT']!r} needs FP16_FIX ('online' "
            "with FP16_FIX_KWARGS={'c': 1/16}, or 'fp32_residual') or an fp16-fix "
            "WEIGHTS file (FP16_FIX='none'): the residual stream exceeds the fp16 range"
        )
    if k["FP16_FIX"] == "weights":
        p.append(
            "FP16_FIX='weights' is not a training fix: train an fp16-fix checkpoint "
            "with FP16_FIX='none' and that file as WEIGHTS (training applies only "
            "online fixes: 'online', 'fp32_residual')"
        )
    return p + component_problems(k)


def component_problems(k: dict) -> list[str]:
    """A single-file WEIGHTS holds only the transformer: the VAE / text encoder the run
    loads need their own knob."""
    if not os.path.isfile(k["WEIGHTS"]):
        return []
    sampling = k["SAMPLE_EVERY"] > 0
    need = {
        "VAE": ("vae", k["LATENT_CACHE"] != "offline" or sampling),
        "TEXT_ENCODER": ("text_encoder", k["TEXT_CACHE"] != "offline" or sampling),
    }
    return [
        f"WEIGHTS={k['WEIGHTS']!r} is a single file: set {name} (e.g. the base "
        f"folder's {sub}/)"
        for name, (sub, needed) in need.items()
        if needed and not k[name]
    ]


def schedule_problems(k: dict) -> list[str]:
    """A SCHEDULE (or sub-schedule) whose warmup never ends before its end."""
    sched = k["SCHEDULE"]
    subs = [sched] if "mode" in sched else list(sched.values())
    names = [""] if "mode" in sched else [f"[{key!r}]" for key in sched]
    p = []
    for name, sub in zip(names, subs):
        raw_end = sub.get("end", -1)
        end = k["STEPS"] if raw_end == -1 else raw_end
        warmup = sub.get("warmup", 0)
        if isinstance(end, int) and warmup >= end:
            bound = "STEPS" if raw_end == -1 else f"SCHEDULE{name}['end']"
            p.append(
                f"SCHEDULE{name} warmup={warmup} >= its end {end} ({bound}): the "
                f"schedule never leaves warmup. Fix: set SCHEDULE{name}['warmup'] below "
                f"{end} (e.g. {max(0, end // 10)}), or raise {bound} above {warmup}"
            )
    return p
