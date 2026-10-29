"""Train-script knobs: collection from globals and validation into one SystemExit. The
area checks live in `training.checks`; this module holds the knob set, the choices,
the precision and memory-plan checks and `check`."""

import torch

from kohakumoss.adapters.build.config import config_from_knobs
from kohakumoss.diffusion.objective import LOSSES, WEIGHTINGS
from kohakumoss.diffusion.timestep import TIMESTEPS
from kohakumoss.encoders.knob_names import ENCODER_FIELDS, RENAMED_SAMPLE, SAMPLE_NAMES
from kohakumoss.logging import LOGGERS
from kohakumoss.memory.knob_map import mixing_problems, plan_fields, primary_names
from kohakumoss.memory.profiles import KNOB_FIELDS, ORDER, PROFILES, resolve
from kohakumoss.memory.release import release_problems
from kohakumoss.offload.accel import default_type
from kohakumoss.precision import COMPUTE
from kohakumoss.training.checks.encode import encode_problems
from kohakumoss.training.checks.encoders import text_problems, vae_problems
from kohakumoss.training.checks.objective import (
    REAL_VARIANTS,
    carries_fp16_fix,
    objective_problems,
    schedule_problems,
)
from kohakumoss.training.checks.outputs import EXPORTS, io_problems
from kohakumoss.training.checks.parallel import (
    THREADED_SCHEDULES,
    dist_backend,
    parallel_problems,
)
from kohakumoss.training.exports import with_export_formats
from kohakumoss.training.layouts import LAYOUTS
from kohakumoss.training.optim.optimizers.factory import OPTIMIZERS
from kohakumoss.training.optim.optimizers.kwargs import optimizer_problems
from kohakumoss.training.stream_loss import spec_of as stream_spec
from kohakumoss.training.stream_loss import stream_loss_problems
from kohakumoss.types.arch import PRESETS
from kohakumoss.types.plans import MemoryPlan
from kohakumoss_common.knobs import choice_problems, unknown_problems
from kohakumoss_data import DataConfig, data_problems
from kohakumoss_kernels import ops as kernel_ops
from kohakumoss_kernels.hardware.arch import arch_of
from kohakumoss_kernels.tuning.tuner import MODES as TUNE_MODES

__all__ = [
    "CHOICES",
    "EXPORTS",
    "KNOWN",
    "REAL_VARIANTS",
    "THREADED_SCHEDULES",
    "carries_fp16_fix",
    "check",
    "dist_backend",
    "plan_problems",
    "validate",
]

KNOWN = frozenset(
    """WEIGHTS VARIANT ARCH_OVERRIDES TRANSFORMER TEXT_ENCODER TEXT_DEVICE TEXT_ENCODE VAE
    VAE_ENCODE VAE_DTYPE ENCODE_DEVICES ENCODE_DEPTH ENCODE_CHUNK TRAIN_TEXT_REFINER DATASETS
    RESOLUTION_MODE TARGET_AREA SIZE_STEP ASPECT_RANGE UPSCALE MAX_TOKENS BATCH_SIZE
    CAPTION TEXT_DROP LATENT_CACHE TEXT_CACHE CACHE_DTYPE CACHE_DIR CAPTION_VARIANTS FLIP
    NUM_WORKERS SEED ALGO PRESET DIM ALPHA ALGO_KWARGS DORA DORA_DETACH DROPOUT
    ADAPTER_MODE TRAIN_NORMS TIMESTEP TIMESTEP_KWARGS LOSS LOSS_KWARGS OPTIMIZER LR
    BETAS EPS WEIGHT_DECAY WD_RULES SCHEDULE GRAD_CLIP EMA PRECISION KEEP_FP32 FP16_FIX
    FP16_FIX_KWARGS
    MEMORY_PROFILE MEMORY_AUTO_FORCE MEMORY_OVERRIDES STEP_RELEASE PARALLEL DIST_BACKEND DEVICES
    PIPELINE_SCHEDULE PIPELINE_SPLIT TRACE_STEPS PROFILE_STEPS FSDP_REDUCE FSDP_RESHARD
    DDP_BACKEND DDP_REDUCE_DTYPE DIST_TIMEOUT DIST_GATE_TIMEOUT DIST_BIND
    CHECKPOINT STREAM STREAM_RESIDENT PINNED ACT_OFFLOAD OPTIM_STATE GRAD_OFFLOAD
    FUSED_STEP BASE_PRECISION FP8_RECIPE FP8_DGRAD INT8_COMPUTE INT8_DGRAD
    MICROBATCHES KERNELS TUNE RETUNE_STEPS COMPILE DRIVER STEPS GRAD_ACC OUTPUT_DIR RUN_NAME LOGGER
    LOG_EVERY METRICS_EVERY PROGRESS SAMPLE_PROMPTS SAMPLE_EVERY SAMPLE_STEPS SAMPLE_CFG SAMPLE_SHIFT
    SAMPLE_SKIP_LAYERS SAMPLE_SIZE SAMPLE_SEEDS VALIDATE_EVERY VALIDATION SAVE_EVERY
    SAVE_MINUTES KEEP_LAST FINAL_CHECKPOINT RESUME EXPORT_FORMATS SAVE_FORMAT SAVE_MERGED SAVE_FUSED_QKV
    CHECKPOINT_EXPORT WEIGHTING OPTIMIZER_KWARGS
    SAMPLE_NEGATIVE PROJECT LORAPLUS_RATIO BATCHING OVERLONG MAX_SAMPLES TOKEN_BUDGET
    CROP PLANNER_KWARGS LATENT_ENCODER_ID LATENT_DRAW
    TEXT_ENCODER_ID BACKGROUND RESAMPLE TEXT_CHARS_PER_TOKEN TEXT_OVERHEAD_TOKENS
    MAX_TEXT_TOKENS PREFETCH PIN_MEMORY NONFINITE_PATIENCE DATA_RESCAN
    DATA_SCAN_WORKERS DATA_WAIT CONCENTRATION_LOSS STREAM_LOSS""".split()
)
CHOICES = {
    "ADAPTER_MODE": ("bypass", "merge"),
    "WD_RULES": ("moss", "none", "all"),
    "MEMORY_PROFILE": (*PROFILES, "auto"),
    "DIST_BACKEND": ("auto", "nccl", "gloo"),
    "TUNE": TUNE_MODES,
    "COMPILE": ("off", "block", "model"),
    "DRIVER": ("native", "lightning"),
    "FP16_FIX": ("none", "online", "fp32_residual"),
    "FSDP_REDUCE": ("step", "micro"),
    "FSDP_RESHARD": ("forward", "backward"),
    "DDP_BACKEND": ("kohakumoss", "torch"),
    "DDP_REDUCE_DTYPE": ("fp32", "param"),
    "DIST_BIND": ("lazy", "device"),
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
RENAMED = {
    **RENAMED_SAMPLE,
    "DIST_STARTUP_TIMEOUT": "renamed to DIST_GATE_TIMEOUT",
    "TEXT_TOKENS_PER_SAMPLE": "removed: MEMORY_PROFILE='auto' reads the dataset's "
    "longest caption",
    **{name: f"the train script's name is SAMPLE_{name}" for name in SAMPLE_NAMES},
    "SIZES": "the train script's name is SAMPLE_SIZE",
    "SIZE": "the train script's name is SAMPLE_SIZE",
}
default_device_type = default_type


def run_name(k: dict) -> str:
    """RUN_NAME, or "run" when empty (output folder, logger run, export file names)."""
    return k["RUN_NAME"] or "run"


def is_full(k: dict) -> bool:
    """Full finetune: the backbone's own weights train."""
    return k["ALGO"] == "full"


def collect(g: dict) -> dict:
    """UPPER_CASE globals of the train script (after the config was applied), with
    MEMORY_PROFILE in its primary spelling (`primary_names`)."""
    return primary_names(
        {k: v for k, v in g.items() if k.isupper() and not k.startswith("_")}
    )


def _registry_ok(name, registry) -> bool:
    return not isinstance(name, str) or name in registry or "." in name


def _unknown(k: dict) -> list[str]:
    return unknown_problems(k, KNOWN, RENAMED, ENCODER_FIELDS)


def _choices(k: dict) -> list[str]:
    return choice_problems(k, CHOICES)


def _cuda_devices(k: dict) -> list[int]:
    """Entries of DEVICES that exist as CUDA devices on this host."""
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return [i for i in k["DEVICES"] if isinstance(i, int) and 0 <= i < count]


def resolve_precision(k: dict) -> str:
    """PRECISION with "auto" resolved: "bf16" when every training CUDA device has it,
    else "fp16"; "bf16" off CUDA."""
    if k["PRECISION"] != "auto":
        return k["PRECISION"]
    if default_device_type() == "cuda":
        bf16 = all(arch_of(f"cuda:{i}").has_bf16 for i in _cuda_devices(k))
        return "bf16" if bf16 else "fp16"
    return "bf16"


def with_precision(k: dict) -> dict:
    """The knobs with PRECISION resolved (`resolve_precision`)."""
    return {**k, "PRECISION": resolve_precision(k)}


def _device_problems(k: dict) -> list[str]:
    """Per-device checks over every entry of DEVICES that exists on this host."""
    if k.get("PRECISION") != "bf16":
        return []
    out = []
    for index in _cuda_devices(k):
        arch = arch_of(f"cuda:{index}")
        if not arch.has_bf16:
            out.append(
                f"PRECISION='bf16' on cuda:{index} ({arch.name}, no bf16): "
                "use PRECISION='fp16'"
            )
    return out


def _value_problems(k: dict) -> list[str]:
    """Names, registries and plain value ranges."""
    p = []
    if k["VARIANT"] not in PRESETS:
        p.append(f"VARIANT={k['VARIANT']!r} not in {sorted(PRESETS)}")
    if not k["WEIGHTS"]:
        p.append("WEIGHTS is empty (a weights root, pipeline dir or single file)")
    if k["ALGO"] not in ALGOS:
        p.append(f"ALGO={k['ALGO']!r} not in {ALGOS}")
    if k["PRECISION"] not in COMPUTE:
        p.append(f"PRECISION={k['PRECISION']!r} not in {(*COMPUTE, 'auto')}")
    for name, reg in (
        ("TIMESTEP", TIMESTEPS),
        ("LOSS", LOSSES),
        ("OPTIMIZER", OPTIMIZERS),
        ("LOGGER", LOGGERS),
        ("PARALLEL", LAYOUTS),
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
    rows = k.get("METRICS_EVERY")
    bad = isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0
    if rows is not None and bad:
        p.append(
            f"METRICS_EVERY must be None (= LOG_EVERY) or an int > 0, got {rows!r}"
        )
    patience = k["NONFINITE_PATIENCE"]
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
        p.append(f"NONFINITE_PATIENCE must be an int >= 0 (0 = off), got {patience!r}")
    if not isinstance(k["MAX_SAMPLES"], int) or k["MAX_SAMPLES"] < 0:
        p.append(
            f"MAX_SAMPLES must be an int >= 0 (0 = no cap), got {k['MAX_SAMPLES']!r}"
        )
    if k["GRAD_CLIP"] < 0:
        p.append("GRAD_CLIP must be >= 0 (0 disables clipping)")
    retune = k.get("RETUNE_STEPS", 0)
    if isinstance(retune, bool) or not isinstance(retune, int) or retune < 0:
        p.append(f"RETUNE_STEPS must be an int >= 0, got {retune!r}")
    p += release_problems(k.get("STEP_RELEASE", "none"))
    return p


def validate(k: dict) -> list[str]:
    """Every problem with the knob set (empty list = valid); PRECISION="auto" is
    resolved first."""
    k = with_precision(k)
    p = _unknown(k) + _choices(k) + _device_problems(k) + _value_problems(k)
    p += data_problems(DataConfig.from_knobs(k))
    p += schedule_problems(k)
    p += objective_problems(k)
    p += _memory_problems(k)
    p += text_problems(k, arch_of)
    p += vae_problems(k)
    p += encode_problems(k, arch_of)
    layout = _layout(k)
    p += parallel_problems(k, layout.problems if layout is not None else None)
    p += optimizer_problems(k)
    p += io_problems(k)
    p += stream_loss_problems(k)
    return list(dict.fromkeys(p))


def memory_overrides(k: dict) -> dict:
    """MEMORY_OVERRIDES, the top-level memory knobs (manual mode) and precision knobs,
    plus the plan fields other knobs own (COMPILE, KERNELS)."""
    return {
        **k["MEMORY_OVERRIDES"],
        **plan_fields(k, k["MEMORY_PROFILE"]),
        "compile": k["COMPILE"],
        "kernels": dict(k["KERNELS"]),
    }


def plan_problems(k: dict, plan: MemoryPlan) -> list[str]:
    """What keeps a resolved MemoryPlan from running with the other knobs (PARALLEL,
    OPTIMIZER, COMPILE). See docs/guides/multi-gpu.md for the supported matrix."""
    p = []
    opt = k["OPTIMIZER"]
    if opt != "adamw" and plan.optim_state != "gpu":
        p.append(
            f"optim_state={plan.optim_state!r} places AdamW states; OPTIMIZER={opt!r} "
            "keeps its own on the GPU: use optim_state='gpu'"
        )
    if opt != "adamw" and plan.stream == "swap":
        p.append(
            f"stream='swap' trains blocks with AdamW, not OPTIMIZER={opt!r}: use a "
            "resident profile (e.g. MEMORY_PROFILE='ckpt-resident') or STREAM='off' "
            "with MEMORY_PROFILE='manual'"
        )
    if plan.fused_step and opt == "muon":
        p.append(
            "OPTIMIZER='muon' needs fused_step=False (per-parameter steps): use a "
            "profile without 'optfused' / stream for full FT (e.g. 'ckpt-resident') or "
            "FUSED_STEP=False with MEMORY_PROFILE='manual'"
        )
    if plan.checkpoint == "sub-block" and stream_spec(k):
        p.append(
            "checkpoint='sub-block' calls block parts directly: STREAM_LOSS / "
            "CONCENTRATION_LOSS need whole-block forwards (CHECKPOINT 'block', "
            "'partial:...' or 'none')"
        )
    if plan.checkpoint == "sub-block" and plan.compile == "block":
        p.append(
            "checkpoint='sub-block' calls block parts directly: not with "
            "COMPILE='block'"
        )
    layout = _layout(k)
    return p + (layout.plan_problems(k, plan) if layout is not None else [])


def _layout(k: dict):
    """The Layout PARALLEL selects, or None when it names none (a knob problem)."""
    name = k["PARALLEL"]
    if not isinstance(name, str) or not _registry_ok(name, LAYOUTS):
        return None
    try:
        return LAYOUTS.get(name)
    except (KeyError, ImportError, AttributeError, ValueError):
        return None


def has_step_state(k: dict) -> bool:
    """The run's adapters carry per-micro-batch step state (full FT carries none)."""
    return not is_full(k) and config_from_knobs(k).has_step_state


def _plans(k: dict) -> tuple[list, list[str]]:
    """Resolved plans of MEMORY_PROFILE (every profile for `auto`) and their errors."""
    overrides = memory_overrides(k)
    names = ORDER if k["MEMORY_PROFILE"] == "auto" else (k["MEMORY_PROFILE"],)
    plans, errors = [], []
    for name in names:
        try:
            plans.append(resolve(name, overrides, default_device_type(), is_full(k)))
        except ValueError as err:
            errors.append(f"MEMORY_PROFILE {name!r}: {err}")
    return plans, errors


def _memory_problems(k: dict) -> list[str]:
    p = mixing_problems(k, k["MEMORY_PROFILE"], "MEMORY_PROFILE")
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


def check(g: dict) -> dict:
    """Knobs from globals (PRECISION resolved, a set EXPORT_FORMATS mapped onto
    SAVE_FORMAT / SAVE_MERGED); raises one SystemExit listing every problem."""
    k = with_export_formats(with_precision(collect(g)))
    problems = validate(k)
    if problems:
        raise SystemExit("invalid configuration:\n  - " + "\n  - ".join(problems))
    return k
