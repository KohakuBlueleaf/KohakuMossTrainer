"""Knob checks of the parallel layout: backend, devices, timeouts, pipeline split and
trace windows for every layout (`parallel_problems`), and the per-layout limits each
`Layout` carries as its `problems(k)` hook (`local_problems`, `ddp_problems`,
`fsdp_problems`, `pipeline_layout_problems`, `threaded_problems`)."""

import sys

from kohakumoss.adapters.build.config import config_from_knobs
from kohakumoss.memory.estimate import base_precision
from kohakumoss.offload.accel import default_type
from kohakumoss.parallel.launch import default_backend
from kohakumoss.parallel.pipeline.distributed import schedule_problems
from kohakumoss.parallel.pipeline.split import SPLITS
from kohakumoss.training.weights import stored_dit_precision
from kohakumoss.types.plans import PrecisionConfig
from kohakumoss_data import DataConfig, pipeline_problems

THREADED_SCHEDULES = ("gpipe", "async")


def dist_backend(k: dict) -> str:
    """DIST_BACKEND with 'auto' resolved the way the launcher resolves it."""
    if k["DIST_BACKEND"] != "auto":
        return k["DIST_BACKEND"]
    return default_backend(default_type())


def ints(value) -> bool:
    """A list / tuple of plain ints (no bools)."""
    return isinstance(value, (list, tuple)) and all(
        isinstance(n, int) and not isinstance(n, bool) for n in value
    )


def _launch_problems(k: dict) -> list[str]:
    p = []
    if k["DIST_BACKEND"] == "nccl" and sys.platform == "win32":
        p.append("DIST_BACKEND='nccl' is unavailable on Windows: use 'gloo' or 'auto'")
    if not k["DEVICES"]:
        p.append("DEVICES is empty")
    for name, default in (("DIST_TIMEOUT", 600), ("DIST_GATE_TIMEOUT", 7200)):
        timeout = k.get(name, default)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            p.append(f"{name} must be a number of seconds > 0, got {timeout!r}")
    module_dropout = float((k.get("DROPOUT") or {}).get("module", 0.0))
    if k.get("DDP_BACKEND") == "torch" and module_dropout > 0:
        p.append(
            "DDP_BACKEND='torch' needs every trainable parameter used in every step: "
            "DROPOUT['module'] > 0 skips adapters; use DDP_BACKEND='kohakumoss'"
        )
    return p


def _windows_problems(k: dict) -> list[str]:
    """A multi-device layout on Windows (multi-GPU training is Linux-only by design)."""
    if sys.platform != "win32":
        return []
    return [
        f"PARALLEL={k['PARALLEL']!r} with DEVICES={tuple(k['DEVICES'])}: multi-GPU "
        "training is Linux-only by design; on Windows use PARALLEL='none' and one "
        "entry in DEVICES"
    ]


def _native_driver(k: dict) -> list[str]:
    if k["DRIVER"] == "native":
        return []
    return [
        f"PARALLEL={k['PARALLEL']!r} runs on DRIVER='native' (DRIVER='lightning' "
        "supports PARALLEL 'none' | 'ddp')"
    ]


def _no_loop_sampling(k: dict) -> list[str]:
    if k["SAMPLE_EVERY"] <= 0 and k["VALIDATE_EVERY"] <= 0:
        return []
    return [
        f"PARALLEL={k['PARALLEL']!r}: sampling / validation during training are not "
        "routed through the pipeline; set SAMPLE_EVERY=0 and VALIDATE_EVERY=0"
    ]


def local_problems(k: dict) -> list[str]:
    """PARALLEL='none': one device."""
    if len(k["DEVICES"]) <= 1:
        return []
    return _windows_problems(k) or [
        "several DEVICES with PARALLEL='none': choose ddp / fsdp / pipeline / threaded"
    ]


def ddp_problems(k: dict) -> list[str]:
    """PARALLEL='ddp'."""
    return _windows_problems(k)


def fsdp_problems(k: dict) -> list[str]:
    """PARALLEL='fsdp': whole-matrix optimizers, fp32 parameters next to 16-bit ones in
    one unit, the gloo backend and rank-0 sampling cannot run on shards."""
    p = _windows_problems(k) + _native_driver(k)
    if k["OPTIMIZER"] == "muon":
        p.append(
            "OPTIMIZER='muon' orthogonalises whole matrices; under PARALLEL='fsdp' it "
            "would see DTensor shards: use PARALLEL='ddp' or another optimizer"
        )
    extra = sorted(set(k["KEEP_FP32"]) - set(PrecisionConfig().keep_fp32))
    if k["OPTIMIZER"] == "adamw16" and extra:
        p.append(
            f"PARALLEL='fsdp' with OPTIMIZER='adamw16': KEEP_FP32 entries {extra} put "
            "fp32 next to 16-bit trainable parameters in one FSDP unit; keep KEEP_FP32 "
            f"within {PrecisionConfig().keep_fp32} or use OPTIMIZER='adamw'"
        )
    if dist_backend(k) == "gloo":
        p.append(
            "PARALLEL='fsdp' needs DIST_BACKEND='nccl': gloo has no PREMUL_SUM "
            "reduce-scatter (see docs/guides/multi-gpu.md)"
        )
    if k["SAMPLE_EVERY"] > 0:
        p.append(
            "PARALLEL='fsdp': sampling runs on rank 0 only and cannot all-gather the "
            "sharded blocks; set SAMPLE_EVERY=0"
        )
    return p


def pipeline_layout_problems(k: dict) -> list[str]:
    """PARALLEL='pipeline': the schedule over the stages, the pipeline's data plan, the
    native driver, no in-loop sampling."""
    p = _windows_problems(k) + _native_driver(k) + _no_loop_sampling(k)
    p += schedule_problems(k["PIPELINE_SCHEDULE"], len(k["DEVICES"]), k["MICROBATCHES"])
    return p + pipeline_problems(DataConfig.from_knobs(k))


def threaded_problems(k: dict) -> list[str]:
    """PARALLEL='threaded': its schedules, the native driver, no in-loop sampling."""
    p = _windows_problems(k) + _native_driver(k) + _no_loop_sampling(k)
    if k["PIPELINE_SCHEDULE"] not in THREADED_SCHEDULES:
        p.append(
            f"PARALLEL='threaded' runs PIPELINE_SCHEDULE in {THREADED_SCHEDULES}, "
            f"got {k['PIPELINE_SCHEDULE']!r}"
        )
    return p


ALLOWED = {
    "fsdp": "checkpointing (CHECKPOINT block / selective / partial:<f>) is supported "
    "there, memory offload is not",
    "pipeline": "checkpointing (every CHECKPOINT) is supported there, memory offload "
    "is not",
    "threaded": "checkpointing (every CHECKPOINT) is supported there, memory offload "
    "is not",
}


def _in_backward_steps(k: dict, plan) -> list[str]:
    kind, p = k["PARALLEL"], []
    allowed = ALLOWED.get(kind, "")
    if plan.fused_step:
        p.append(
            f"PARALLEL={kind!r} needs fused_step=False (none / ddp only): {allowed}"
        )
    if plan.grad_offload:
        p.append(
            f"PARALLEL={kind!r} needs grad_offload=False (none / ddp only): {allowed}"
        )
    return p


def _resident(k: dict, plan) -> list[str]:
    kind, p = k["PARALLEL"], []
    allowed = ALLOWED.get(kind, "")
    precision = base_precision(plan, stored_dit_precision(k))
    for field, value, ok in (
        ("stream", plan.stream, "off"),
        ("optim_state", plan.optim_state, "gpu"),
        ("base_precision", precision, "compute"),
        ("act_offload", plan.act_offload, False),
    ):
        if value != ok:
            p.append(
                f"PARALLEL={kind!r} needs {field}={ok!r}, not {value!r}: {allowed}"
            )
    return p


def ddp_plan_problems(k: dict, plan) -> list[str]:
    """PARALLEL='ddp' with a memory plan: torch DDP reduces device-resident gradients."""
    torch_ddp = k.get("DDP_BACKEND") == "torch"
    if torch_ddp and (plan.fused_step or plan.grad_offload or plan.stream != "off"):
        return [
            "DDP_BACKEND='torch' reduces gradients only where they stay on the device "
            "until the step: it needs fused_step=False, grad_offload=False and "
            f"stream='off' (this plan: fused_step={plan.fused_step}, grad_offload="
            f"{plan.grad_offload}, stream={plan.stream!r}); use DDP_BACKEND='kohakumoss'"
        ]
    return []


def fsdp_plan_problems(k: dict, plan) -> list[str]:
    """PARALLEL='fsdp' with a memory plan: resident, per-step optimizer, whole blocks."""
    p = _in_backward_steps(k, plan) + _resident(k, plan)
    if plan.checkpoint == "sub-block":
        p.append(
            "checkpoint='sub-block' calls block parts directly: not with "
            "PARALLEL='fsdp'"
        )
    return p


def pipeline_plan_problems(k: dict, plan) -> list[str]:
    """PARALLEL='pipeline' with a memory plan: resident, per-step optimizer."""
    return _in_backward_steps(k, plan) + _resident(k, plan)


def threaded_plan_problems(k: dict, plan) -> list[str]:
    """PARALLEL='threaded' with a memory plan: resident, per-step optimizer, and no
    per-micro-batch adapter state under the async schedule's recompute."""
    p = _in_backward_steps(k, plan) + _resident(k, plan)
    stepped = k["ALGO"] != "full" and config_from_knobs(k).has_step_state
    async_recompute = k["PIPELINE_SCHEDULE"] == "async" and plan.checkpoint != "none"
    if async_recompute and stepped:
        p.append(
            f"ALGO={k['ALGO']!r} / DROPOUT['module'] draw per-micro-batch adapter state "
            "that the async threaded pipeline's backward recompute cannot see (the "
            "next forward has replaced it): use PIPELINE_SCHEDULE='gpipe' or "
            "checkpoint='none'"
        )
    return p


def _pipeline_problems(k: dict) -> list[str]:
    p = []
    split = k["PIPELINE_SPLIT"]
    counts = ints(split) and len(split) > 0 and all(n > 0 for n in split)
    if split not in SPLITS and not counts:
        p.append(
            f"PIPELINE_SPLIT={split!r}: one of {SPLITS} or layers per stage, "
            "e.g. (11, 10, 11, 10)"
        )
    for name in ("TRACE_STEPS", "PROFILE_STEPS"):
        trace = k[name]
        window = (
            ints(trace) and len(trace) == 2 and 1 <= trace[0] < trace[1] <= k["STEPS"]
        )
        if not (window or (ints(trace) and len(trace) == 0)):
            p.append(
                f"{name}={trace!r}: () or (first, last) with "
                "1 <= first < last <= STEPS"
            )
    return p


def parallel_problems(k: dict, layout_problems=None) -> list[str]:
    """Every parallel knob problem: the layout-independent launch and pipeline-split
    checks, then the selected layout's own `problems(k)` hook (`layout_problems`)."""
    own = layout_problems(k) if layout_problems is not None else []
    return _launch_problems(k) + _pipeline_problems(k) + own
