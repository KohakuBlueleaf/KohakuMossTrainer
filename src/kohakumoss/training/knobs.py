"""Train-script knobs: collection from globals and validation into one SystemExit."""

import difflib
import os
import sys
from pathlib import Path

import torch

from kohakumoss.diffusion.objective import LOSSES, WEIGHTINGS
from kohakumoss.diffusion.timestep import TIMESTEPS
from kohakumoss.logging import LOGGERS
from kohakumoss.offload.accel import default_type
from kohakumoss.parallel.launch import default_backend
from kohakumoss.training.memory.profiles import KNOB_FIELDS, ORDER, resolve
from kohakumoss.training.optim import factory as optim_factory
from kohakumoss.training.optim.factory import OPTIMIZERS
from kohakumoss.training.precision import DTYPES
from kohakumoss.types.arch import PRESETS
from kohakumoss.types.plans import MemoryPlan
from kohakumoss_data import DataConfig, data_problems, pipeline_problems
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
    LYCORIS_COMPILE_DYNAMIC CROP PLANNER_KWARGS LATENT_ENCODER_ID LATENT_DRAW
    TEXT_ENCODER_ID BACKGROUND RESAMPLE TEXT_CHARS_PER_TOKEN TEXT_OVERHEAD_TOKENS
    MAX_TEXT_TOKENS PREFETCH PIN_MEMORY""".split()
)
OPTIONAL = frozenset()
CHOICES = {
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
THREADED_SCHEDULES = ("gpipe", "async")


def run_name(k: dict) -> str:
    """RUN_NAME, or "run" when empty (output folder, logger run, export file names)."""
    return k["RUN_NAME"] or "run"


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


def _cuda_devices(k: dict) -> list[int]:
    """Entries of DEVICES that exist as CUDA devices on this host."""
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return [i for i in k["DEVICES"] if isinstance(i, int) and 0 <= i < count]


def resolve_precision(k: dict) -> str:
    """PRECISION with "auto" resolved: "fp32" on CPU (the reference mode), else "bf16"
    when every training device has it, else "fp16"."""
    if k["PRECISION"] != "auto":
        return k["PRECISION"]
    match default_device_type():
        case "cpu":
            return "fp32"
        case "cuda":
            bf16 = all(arch_of(f"cuda:{i}").has_bf16 for i in _cuda_devices(k))
            return "bf16" if bf16 else "fp16"
    return "bf16"


def with_precision(k: dict) -> dict:
    """The knobs with PRECISION resolved (`resolve_precision`)."""
    return {**k, "PRECISION": resolve_precision(k)}


def _device_rules(k: dict) -> list[str]:
    """Per-device checks over every entry of DEVICES that exists on this host."""
    out = []
    if k.get("PRECISION") == "fp32" and default_device_type() != "cpu":
        out.append(
            "PRECISION='fp32' trains with fp32 matmuls, which GPU training does not "
            "use: set 'bf16', 'fp16' or 'auto' (fp32 is the CPU / test reference mode)"
        )
    if k.get("PRECISION") != "bf16":
        return out
    for index in _cuda_devices(k):
        arch = arch_of(f"cuda:{index}")
        if not arch.has_bf16:
            out.append(
                f"PRECISION='bf16' on cuda:{index} ({arch.name}, no bf16): "
                "use PRECISION='fp16'"
            )
    return out


REAL_VARIANTS = ("base", "sft")


def _objective_rules(k: dict) -> list[str]:
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
    ):
        p.append(
            f"PRECISION='fp16' on VARIANT={k['VARIANT']!r} needs FP16_FIX ('online' "
            "with FP16_FIX_KWARGS={'c': 1/16}, 'weights' or 'fp32_residual'): the "
            "residual stream exceeds the fp16 range"
        )
    return p


def validate(k: dict) -> list[str]:
    """Every problem with the knob set (empty list = valid); PRECISION="auto" is
    resolved first."""
    k = with_precision(k)
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
    p += data_problems(DataConfig.from_knobs(k))
    p += _schedule_rules(k)
    p += _objective_rules(k)
    p += _memory_rules(k)
    p += _parallel_rules(k)
    p += _optimizer_rules(k)
    p += _io_rules(k)
    return list(dict.fromkeys(p))


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


default_device_type = default_type


def dist_backend(k: dict) -> str:
    """DIST_BACKEND with 'auto' resolved the way the launcher resolves it."""
    if k["DIST_BACKEND"] != "auto":
        return k["DIST_BACKEND"]
    return default_backend(default_device_type())


def memory_overrides(k: dict) -> dict:
    """MEMORY_OVERRIDES plus the plan fields other knobs own (COMPILE, KERNELS)."""
    return {
        **k["MEMORY_OVERRIDES"],
        "compile": k["COMPILE"],
        "kernels": dict(k["KERNELS"]),
    }


def plan_problems(k: dict, plan: MemoryPlan) -> list[str]:
    """What keeps a resolved MemoryPlan from running with the other knobs (PARALLEL,
    OPTIMIZER, COMPILE). See docs/guides/multi-gpu.md for the supported matrix."""
    p = []
    kind, opt = k["PARALLEL"], k["OPTIMIZER"]
    if opt != "adamw" and plan.optim_state != "gpu":
        p.append(
            f"optim_state={plan.optim_state!r} places AdamW states; OPTIMIZER={opt!r} "
            "keeps its own on the GPU: use optim_state='gpu'"
        )
    if opt == "adamw" and plan.optim_state == "8bit" and optim_factory.bnb is None:
        p.append(
            "optim_state='8bit' needs the bitsandbytes package: install it or set "
            "optim_state 'gpu' / 'stream' / 'cpu'"
        )
    if opt != "adamw" and plan.stream == "swap":
        p.append(f"stream='swap' trains blocks with AdamW, not OPTIMIZER={opt!r}")
    if plan.fused_step and opt == "muon":
        p.append("OPTIMIZER='muon' needs fused_step=False (per-parameter steps)")
    if kind != "none" and plan.fused_step:
        p.append("fused_step is single-device: set fused_step=False")
    if kind != "none" and plan.grad_offload:
        p.append("grad_offload is single-device: set grad_offload=False")
    if kind == "ddp" and plan.stream == "swap":
        p.append("PARALLEL='ddp' does not reduce streamed (swap) block gradients")
    if kind in ("fsdp", "pipeline", "threaded"):
        for field, value, ok in (
            ("stream", plan.stream, "off"),
            ("optim_state", plan.optim_state, "gpu"),
            ("base_precision", plan.base_precision, "compute"),
            ("act_offload", plan.act_offload, False),
        ):
            if value != ok:
                p.append(f"PARALLEL={kind!r} needs {field}={ok!r}, not {value!r}")
    if plan.checkpoint == "sub-block" and (kind == "fsdp" or plan.compile == "block"):
        p.append(
            "checkpoint='sub-block' calls block parts directly: not with "
            "PARALLEL='fsdp' or COMPILE='block'"
        )
    return p


def _plans(k: dict) -> tuple[list, list[str]]:
    """Resolved plans of MEMORY_PROFILE (every profile for `auto`) and their errors."""
    overrides = memory_overrides(k)
    names = ORDER if k["MEMORY_PROFILE"] == "auto" else (k["MEMORY_PROFILE"],)
    plans, errors = [], []
    for name in names:
        try:
            plans.append(
                resolve(name, overrides, default_device_type(), k["ALGO"] == "full")
            )
        except ValueError as err:
            errors.append(f"MEMORY_PROFILE {name!r}: {err}")
    return plans, errors


def _memory_rules(k: dict) -> list[str]:
    p = []
    known = {f for f in MemoryPlan.__dataclass_fields__}
    bad = sorted(set(k["MEMORY_OVERRIDES"]) - known)
    if bad:
        p.append(f"MEMORY_OVERRIDES has unknown fields {bad}; valid: {sorted(known)}")
    for field, knob in KNOB_FIELDS.items():
        if field in k["MEMORY_OVERRIDES"]:
            p.append(f"MEMORY_OVERRIDES {field!r}: set the {knob} knob instead")
    if not bad and k["MEMORY_PROFILE"] in CHOICES["MEMORY_PROFILE"]:
        plans, errors = _plans(k)
        runnable = [q for q in plans if not plan_problems(k, q)]
        if k["MEMORY_PROFILE"] != "auto":
            p += errors + [m for q in plans for m in plan_problems(k, q)]
        elif not runnable:
            p.append(
                "MEMORY_PROFILE='auto': no profile runs with these knobs: "
                + "; ".join(errors + [m for q in plans for m in plan_problems(k, q)])
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
    multi = k["PARALLEL"] != "none" or len(k["DEVICES"]) > 1
    if multi and sys.platform == "win32":
        p.append(
            f"PARALLEL={k['PARALLEL']!r} with DEVICES={tuple(k['DEVICES'])}: multi-GPU "
            "training is Linux-only by design; on Windows use PARALLEL='none' and one "
            "entry in DEVICES"
        )
    elif k["PARALLEL"] == "none" and len(k["DEVICES"]) > 1:
        p.append(
            "several DEVICES with PARALLEL='none': choose ddp / fsdp / pipeline / threaded"
        )
    if k["PARALLEL"] == "pipeline" and k["MICROBATCHES"] < len(k["DEVICES"]):
        p.append("PARALLEL='pipeline' needs MICROBATCHES >= number of stages")
    if k["PARALLEL"] == "pipeline":
        p += pipeline_problems(DataConfig.from_knobs(k))
    if k["PARALLEL"] == "threaded" and k["PIPELINE_SCHEDULE"] not in THREADED_SCHEDULES:
        p.append(
            f"PARALLEL='threaded' runs PIPELINE_SCHEDULE in {THREADED_SCHEDULES}, "
            f"got {k['PIPELINE_SCHEDULE']!r}"
        )
    if k["PARALLEL"] == "fsdp" and k["OPTIMIZER"] == "muon":
        p.append(
            "OPTIMIZER='muon' orthogonalises whole matrices; under PARALLEL='fsdp' it "
            "would see DTensor shards: use PARALLEL='ddp' or another optimizer"
        )
    if k["PARALLEL"] == "fsdp" and dist_backend(k) == "gloo":
        p.append(
            "PARALLEL='fsdp' needs DIST_BACKEND='nccl': gloo has no PREMUL_SUM "
            "reduce-scatter (see docs/guides/multi-gpu.md)"
        )
    if k["PARALLEL"] == "fsdp" and k["SAMPLE_EVERY"] > 0:
        p.append(
            "PARALLEL='fsdp': sampling runs on rank 0 only and cannot all-gather the "
            "sharded blocks; set SAMPLE_EVERY=0"
        )
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
    return p


def check(g: dict) -> dict:
    """Knobs from globals (PRECISION resolved); raises one SystemExit listing every
    problem."""
    k = with_precision(collect(g))
    problems = validate(k)
    if problems:
        raise SystemExit("invalid configuration:\n  - " + "\n  - ".join(problems))
    return k
