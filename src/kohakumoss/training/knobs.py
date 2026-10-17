"""Train-script knobs: collection from globals and validation into one SystemExit."""

import difflib
import os
import sys
from pathlib import Path

import torch

from kohakumoss.diffusion.objective import LOSSES, WEIGHTINGS
from kohakumoss.diffusion.timestep import TIMESTEPS
from kohakumoss.logging import LOGGERS
from kohakumoss.training.memory.profiles import PROFILES
from kohakumoss.training.optim.factory import OPTIMIZERS
from kohakumoss.training.precision import DTYPES
from kohakumoss.types.arch import PRESETS
from kohakumoss.types.plans import MemoryPlan
from kohakumoss_data import DataConfig, pipeline_problems
from kohakumoss_kernels import ops as kernel_ops
from kohakumoss_kernels.hardware.arch import arch_of

KNOWN = frozenset(
    """WEIGHTS VARIANT ARCH_OVERRIDES TEXT_ENCODER TEXT_DEVICE VAE TRAIN_TEXT_REFINER DATASETS
    RESOLUTION_MODE TARGET_AREA SIZE_STEP ASPECT_RANGE UPSCALE MAX_TOKENS BATCH_SIZE
    CAPTION TEXT_DROP LATENT_CACHE TEXT_CACHE CACHE_DTYPE CACHE_DIR CAPTION_VARIANTS FLIP
    NUM_WORKERS SEED ALGO PRESET DIM ALPHA ALGO_KWARGS DORA DORA_DETACH DROPOUT
    ADAPTER_MODE TRAIN_NORMS TIMESTEP TIMESTEP_KWARGS LOSS LOSS_KWARGS OPTIMIZER LR
    BETAS EPS WEIGHT_DECAY WD_RULES SCHEDULE GRAD_CLIP EMA PRECISION KEEP_FP32 FP16_FIX
    FP16_FIX_KWARGS
    MEMORY_PROFILE MEMORY_OVERRIDES PARALLEL DIST_BACKEND DEVICES PIPELINE_SCHEDULE
    MICROBATCHES KERNELS TUNE COMPILE DRIVER STEPS GRAD_ACC OUTPUT_DIR RUN_NAME LOGGER
    LOG_EVERY SAMPLE_PROMPTS SAMPLE_EVERY SAMPLE_STEPS SAMPLE_CFG SAMPLE_SHIFT
    SAMPLE_SKIP_LAYERS SAMPLE_SIZE SAMPLE_SEEDS VALIDATE_EVERY VALIDATION SAVE_EVERY
    SAVE_MINUTES KEEP_LAST RESUME EXPORT_FORMATS WEIGHTING OPTIMIZER_KWARGS FULL_MODE
    SAMPLE_NEGATIVE PROJECT LORAPLUS_RATIO BATCHING OVERLONG MAX_SAMPLES TOKEN_BUDGET
    LYCORIS_COMPILE_DYNAMIC""".split()
)
OPTIONAL = frozenset()
CHOICES = {
    "RESOLUTION_MODE": ("resize", "bucket", "shifted_square", "square_crop"),
    "BATCHING": ("tokens", "tokens_ffd", "count"),
    "OVERLONG": ("drop", "truncate_text", "drop_tokens"),
    "TOKEN_BUDGET": ("estimate", "exact"),
    "UPSCALE": ("never", "to_area"),
    "LATENT_CACHE": ("off", "offline", "online"),
    "TEXT_CACHE": ("off", "offline", "online"),
    "CACHE_DTYPE": ("bf16", "fp8"),
    "ADAPTER_MODE": ("bypass", "merge"),
    "WD_RULES": ("moss", "none", "all"),
    "MEMORY_PROFILE": ("speed", "balanced", "low", "minimal", "auto"),
    "PARALLEL": ("none", "ddp", "fsdp", "pipeline", "threaded"),
    "DIST_BACKEND": ("auto", "nccl", "gloo"),
    "TUNE": ("auto", "off", "prewarm"),
    "COMPILE": ("off", "block", "model"),
    "DRIVER": ("native", "lightning"),
    "FP16_FIX": ("none", "weights", "online", "fp32_residual"),
}
ALGOS = (
    "lora",
    "locon",
    "loha",
    "lokr",
    "oft",
    "boft",
    "glora",
    "ia3",
    "dylora",
    "tlora",
    "full",
)
EXPORTS = ("comfy", "lycoris", "merged", "diffusers")


def collect(g: dict) -> dict:
    """UPPER_CASE globals of the train script (after the config was applied)."""
    return {k: v for k, v in g.items() if k.isupper() and not k.startswith("_")}


def _registry_ok(name, registry) -> bool:
    return not isinstance(name, str) or name in registry or "." in name


def _unknown(k: dict) -> list[str]:
    out = []
    for name in sorted(set(k) - KNOWN - OPTIONAL):
        close = difflib.get_close_matches(name, KNOWN | OPTIONAL, n=1)
        hint = f" (did you mean {close[0]}?)" if close else ""
        out.append(f"unknown knob {name}{hint}")
    return out


def _choices(k: dict) -> list[str]:
    return [
        f"{name}={k[name]!r} not in {opts}"
        for name, opts in CHOICES.items()
        if name in k and k[name] not in opts
    ]


def _device_rules(k: dict) -> list[str]:
    out = []
    if not torch.cuda.is_available():
        return out
    arch = arch_of("cuda")
    if k.get("PRECISION") == "bf16" and not arch.has_bf16:
        out.append(f"PRECISION='bf16' on {arch.name} (no bf16): use PRECISION='fp16'")
    return out


def validate(k: dict) -> list[str]:
    """Every problem with the knob set (empty list = valid)."""
    p = _unknown(k) + _choices(k) + _device_rules(k)
    if k["VARIANT"] not in PRESETS:
        p.append(f"VARIANT={k['VARIANT']!r} not in {sorted(PRESETS)}")
    if not k["WEIGHTS"]:
        p.append("WEIGHTS is empty (a weights path, or 'random' for smoke runs)")
    if k["ALGO"] not in ALGOS:
        p.append(f"ALGO={k['ALGO']!r} not in {ALGOS}")
    if k["PRECISION"] not in DTYPES:
        p.append(f"PRECISION={k['PRECISION']!r} not in {tuple(DTYPES)}")
    for name, reg in (
        ("TIMESTEP", TIMESTEPS),
        ("LOSS", LOSSES),
        ("OPTIMIZER", OPTIMIZERS),
        ("LOGGER", LOGGERS),
    ):
        if not _registry_ok(k[name], reg):
            p.append(f"{name}={k[name]!r} not in {reg.names()}")
    weighting = k.get("WEIGHTING", "none")
    if not _registry_ok(weighting, WEIGHTINGS):
        p.append(f"WEIGHTING={weighting!r} not in {WEIGHTINGS.names()}")
    if k["LR"] <= 0:
        p.append(f"LR must be positive, got {k['LR']}")
    if k["STEPS"] <= 0 or k["GRAD_ACC"] < 1:
        p.append("STEPS must be > 0 and GRAD_ACC >= 1")
    if k["LOG_EVERY"] <= 0:
        p.append("LOG_EVERY must be > 0")
    if not isinstance(k["MAX_SAMPLES"], int) or k["MAX_SAMPLES"] < 0:
        p.append(
            f"MAX_SAMPLES must be an int >= 0 (0 = no cap), got {k['MAX_SAMPLES']!r}"
        )
    if k["GRAD_CLIP"] < 0:
        p.append("GRAD_CLIP must be >= 0 (0 disables clipping)")
    p += _schedule_rules(k)
    p += _memory_rules(k)
    p += _parallel_rules(k)
    p += _optimizer_rules(k)
    p += _io_rules(k)
    return p


def _schedule_rules(k: dict) -> list[str]:
    sched = k["SCHEDULE"]
    subs = [sched] if "mode" in sched else list(sched.values())
    p = []
    for sub in subs:
        end = sub.get("end", -1)
        end = k["STEPS"] if end == -1 else end
        if isinstance(end, int) and sub.get("warmup", 0) >= end:
            p.append(
                f"SCHEDULE warmup {sub.get('warmup')} >= end {end} (STEPS={k['STEPS']})"
            )
    return p


def _memory_rules(k: dict) -> list[str]:
    p = []
    known = {f for f in MemoryPlan.__dataclass_fields__}
    bad = sorted(set(k["MEMORY_OVERRIDES"]) - known)
    if bad:
        p.append(f"MEMORY_OVERRIDES has unknown fields {bad}; valid: {sorted(known)}")
    profile = PROFILES.get(k["MEMORY_PROFILE"], {})
    fused = k["MEMORY_OVERRIDES"].get("fused_step", profile.get("fused_step", False))
    if fused and k["PARALLEL"] != "none":
        p.append(
            "fused_step (per-parameter steps in backward) is single-device: set MEMORY_OVERRIDES fused_step=False"
        )
    if fused and k["OPTIMIZER"] == "muon":
        p.append(
            "fused_step steps parameters one at a time; OPTIMIZER='muon' needs fused_step=False"
        )
    offload = k["MEMORY_OVERRIDES"].get(
        "grad_offload", profile.get("grad_offload", False)
    )
    if offload and k["PARALLEL"] != "none":
        p.append(
            "grad_offload keeps gradients on the host and is single-device: "
            "set MEMORY_OVERRIDES grad_offload=False"
        )
    bad_ops = sorted(set(k["KERNELS"]) - set(kernel_ops.ALL) - {"lycoris"})
    if bad_ops:
        p.append(f"KERNELS has unknown ops {bad_ops}")
    if k["FP16_FIX"] == "online" and "c" not in k["FP16_FIX_KWARGS"]:
        p.append(
            "FP16_FIX='online' needs FP16_FIX_KWARGS['c'] (e.g. 1/16 for the base)"
        )
    return p


def _parallel_rules(k: dict) -> list[str]:
    p = []
    if k["DIST_BACKEND"] == "nccl" and sys.platform == "win32":
        p.append("DIST_BACKEND='nccl' is unavailable on Windows: use 'gloo' or 'auto'")
    if k["DRIVER"] == "lightning" and k["PARALLEL"] not in ("none", "ddp"):
        p.append(
            "DRIVER='lightning' supports PARALLEL 'none' | 'ddp'; use DRIVER='native'"
        )
    if not k["DEVICES"]:
        p.append("DEVICES is empty")
    if k["PARALLEL"] == "none" and len(k["DEVICES"]) > 1:
        p.append(
            "several DEVICES with PARALLEL='none': choose ddp / fsdp / pipeline / threaded"
        )
    if k["PARALLEL"] == "pipeline" and k["MICROBATCHES"] < len(k["DEVICES"]):
        p.append("PARALLEL='pipeline' needs MICROBATCHES >= number of stages")
    if k["PARALLEL"] in ("pipeline", "threaded") and sys.platform == "win32":
        p.append(
            f"PARALLEL={k['PARALLEL']!r} is Linux-only (pipeline stages need NCCL "
            "point-to-point / CUDA peer streams); on Windows use 'ddp' or 'none'"
        )
    if k["PARALLEL"] == "pipeline":
        p += pipeline_problems(DataConfig.from_knobs(k))
    if k["PARALLEL"] in ("pipeline", "threaded") and k["DRIVER"] != "native":
        p.append(f"PARALLEL={k['PARALLEL']!r} runs on DRIVER='native'")
    if k["PARALLEL"] in ("pipeline", "threaded") and (
        k["SAMPLE_EVERY"] > 0 or k["VALIDATE_EVERY"] > 0
    ):
        p.append(
            f"PARALLEL={k['PARALLEL']!r}: sampling / validation during training are not "
            "routed through the pipeline; set SAMPLE_EVERY=0 and VALIDATE_EVERY=0"
        )
    return p


def _optimizer_rules(k: dict) -> list[str]:
    p = []
    opt = k["OPTIMIZER"]
    if opt == "adamw16" and k["PRECISION"] == "fp32":
        p.append(
            "OPTIMIZER='adamw16' keeps 16-bit params: PRECISION must be fp16 or bf16"
        )
    rounding = k.get("OPTIMIZER_KWARGS", {}).get("rounding", "nearest")
    if rounding == "stochastic" and k["PRECISION"] != "bf16":
        p.append("stochastic rounding is bf16-only: use rounding='kahan' for fp16")
    ema = k["EMA"]
    if not (ema in (None, 0, 0.0) or isinstance(ema, (int, float, list, tuple, dict))):
        p.append(
            f"EMA={ema!r}: 0 | decay | [decays] | {{'kind': 'power', 'sigma_rel': [...]}}"
        )
    if isinstance(ema, (int, float)) and ema and not 0 < ema < 1:
        p.append(f"EMA decay must be in (0, 1), got {ema}")
    return p


def _io_rules(k: dict) -> list[str]:
    p = []
    bad = [f for f in k["EXPORT_FORMATS"] if f not in EXPORTS]
    if bad:
        p.append(f"EXPORT_FORMATS has unknown formats {bad}; valid {EXPORTS}")
    if (
        k["ALGO"] == "full"
        and set(k["EXPORT_FORMATS"]) & {"comfy", "lycoris"}
        and k.get("FULL_MODE", "direct") == "direct"
    ):
        p.append(
            "direct full finetune exports 'merged' / 'diffusers', not adapter formats"
        )
    if k["SAMPLE_EVERY"] > 0 and not k["SAMPLE_PROMPTS"]:
        p.append("SAMPLE_EVERY > 0 needs SAMPLE_PROMPTS")
    if k["VALIDATE_EVERY"] > 0 and not k["VALIDATION"].get("datasets"):
        p.append("VALIDATE_EVERY > 0 needs VALIDATION['datasets']")
    if k["SAVE_EVERY"] < 0 or k["KEEP_LAST"] < 0:
        p.append("SAVE_EVERY and KEEP_LAST must be >= 0")
    resume = k["RESUME"]
    if resume not in ("", "latest") and not Path(resume).exists():
        p.append(f"RESUME={resume!r} does not exist")
    weights = k["WEIGHTS"]
    if weights not in ("", "random") and not os.path.exists(weights):
        p.append(f"WEIGHTS={weights!r} does not exist")
    if not k["DATASETS"]:
        p.append("DATASETS is empty")
    return p


def check(g: dict) -> dict:
    """Knobs from globals; raises one SystemExit listing every problem."""
    k = collect(g)
    problems = validate(k)
    if problems:
        raise SystemExit("invalid configuration:\n  - " + "\n  - ".join(problems))
    return k
