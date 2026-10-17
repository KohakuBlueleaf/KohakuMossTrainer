"""Memory profiles: named presets over MemoryPlan knobs, resolved once per run.

See docs/internals/offload.md for what each knob trades.
"""

from dataclasses import fields, replace

import torch

from kohakumoss.types.plans import MemoryPlan
from kohakumoss_kernels.hardware.arch import arch_of

CHECKPOINT_MODES = ("none", "block", "sub-block", "selective")
STREAM_MODES = ("off", "ring", "swap")
BASE_PRECISIONS = ("compute", "fp8", "int8", "quant")
KNOB_FIELDS = {"compile": "COMPILE", "kernels": "KERNELS"}
OPTIM_STATES = ("gpu", "stream", "8bit", "cpu")
COMPILE_MODES = ("off", "block", "model")

PROFILES: dict[str, dict] = {
    "speed": dict(
        checkpoint="none",
        act_offload=False,
        base_precision="compute",
        stream="off",
        optim_state="gpu",
        fused_step=False,
    ),
    "balanced": dict(
        checkpoint="block",
        act_offload=False,
        base_precision="compute",
        stream="off",
        optim_state="gpu",
        fused_step=False,
    ),
    "low": dict(
        checkpoint="block",
        act_offload=True,
        base_precision="quant",
        stream="off",
        optim_state="8bit",
        fused_step=True,
    ),
    "minimal": dict(
        checkpoint="block",
        act_offload=True,
        base_precision="quant",
        stream="ring",
        stream_resident=2,
        optim_state="stream",
        fused_step=True,
        grad_offload=True,
    ),
}
ORDER = ("speed", "balanced", "low", "minimal")


def checkpoint_mode(value: str) -> tuple[str, float]:
    """`"partial:0.5"` -> ("block", 0.5): checkpoint that fraction of blocks; else 1.0."""
    if value.startswith("partial:"):
        return "block", float(value.split(":", 1)[1])
    return value, 1.0 if value != "none" else 0.0


def resolve_base_precision(value: str, device: torch.device | str) -> str:
    """`quant` picks fp8 where the arch has fp8 GEMMs, else int8."""
    if value != "quant":
        return value
    arch = arch_of(device)
    return "fp8" if arch.has_fp8 else "int8"


def validate(
    plan: MemoryPlan,
    trainable_base: bool = False,
    inference: bool = False,
    device: torch.device | str | None = None,
) -> list[str]:
    """Every problem with `plan`, as messages (empty = valid)."""
    problems = []
    if (
        device is not None
        and plan.base_precision == "fp8"
        and not arch_of(device).has_fp8
    ):
        problems.append(
            f"base_precision 'fp8' needs fp8 GEMMs (sm89+), not {arch_of(device).name}"
        )
    if plan.stream == "swap" and not trainable_base and not inference:
        problems.append(
            "stream='swap' streams trainable blocks; a frozen base uses 'ring'"
        )
    mode, frac = checkpoint_mode(plan.checkpoint)
    if mode not in CHECKPOINT_MODES:
        problems.append(f"checkpoint {plan.checkpoint!r} not in {CHECKPOINT_MODES}")
    if not 0.0 <= frac <= 1.0:
        problems.append(f"checkpoint fraction {frac} outside [0, 1]")
    if plan.stream not in STREAM_MODES:
        problems.append(f"stream {plan.stream!r} not in {STREAM_MODES}")
    if not inference and plan.stream != "off" and (mode == "none" or frac < 1.0):
        problems.append("weight streaming needs every block checkpointed")
    if plan.stream != "off" and plan.stream_resident < 2:
        problems.append("weight streaming needs stream_resident >= 2")
    if plan.stream == "ring" and trainable_base:
        problems.append("stream='ring' is frozen-base only; use 'swap' for full FT")
    if plan.base_precision not in BASE_PRECISIONS:
        problems.append(
            f"base_precision {plan.base_precision!r} not in {BASE_PRECISIONS}"
        )
    if plan.base_precision in ("fp8", "int8", "quant") and trainable_base:
        problems.append("a quantized base cannot be trained (full FT)")
    if plan.optim_state not in OPTIM_STATES:
        problems.append(f"optim_state {plan.optim_state!r} not in {OPTIM_STATES}")
    swap = plan.stream == "swap" and trainable_base and not inference
    if swap and plan.fused_step != (plan.optim_state != "cpu"):
        problems.append(
            "stream='swap' steps blocks inside backward with a GPU optim_state "
            "(fused_step=True) and after backward with optim_state='cpu' "
            "(fused_step=False)"
        )
    if plan.compile not in COMPILE_MODES:
        problems.append(f"compile {plan.compile!r} not in {COMPILE_MODES}")
    return problems


def resolve(
    profile: str,
    overrides: dict | None = None,
    device: torch.device | str = "cpu",
    trainable_base: bool = False,
    inference: bool = False,
) -> MemoryPlan:
    """Named profile + overrides -> concrete MemoryPlan for `device`; raises listing all
    problems. `auto` is resolved by `planner.auto_plan`, not here. With `inference`,
    activation knobs are dropped (no_grad saves nothing) and streaming needs no
    checkpointing."""
    if profile not in PROFILES:
        raise ValueError(f"memory profile {profile!r} not in {tuple(PROFILES)}")
    overrides = dict(overrides or {})
    known = {f.name for f in fields(MemoryPlan)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"unknown MemoryPlan fields: {unknown}")
    defaults = {k: v for k, v in PROFILES[profile].items() if k in known}
    values = {**defaults, **overrides, "profile": profile}
    if trainable_base and "base_precision" not in overrides:
        values["base_precision"] = "compute"
    if trainable_base and "stream" not in overrides and values.get("stream") == "ring":
        values["stream"] = "swap"
    swap = trainable_base and not inference and values.get("stream") == "swap"
    gpu_optim = values.get("optim_state", "gpu") != "cpu"
    if swap and "fused_step" not in overrides:
        values["fused_step"] = gpu_optim
    if values.get("grad_offload") and gpu_optim and "fused_step" not in overrides:
        values["fused_step"] = True
    if inference:
        values.update(checkpoint="none", act_offload=False, fused_step=False)
        values.pop("grad_offload", None)
        values["optim_state"] = "gpu"
        if values.get("stream") == "swap":
            values["stream"] = "ring"
    plan = replace(MemoryPlan(), **values)
    plan = replace(
        plan, base_precision=resolve_base_precision(plan.base_precision, device)
    )
    problems = validate(plan, trainable_base and not inference, inference, device)
    if problems:
        raise ValueError("invalid memory plan: " + "; ".join(problems))
    return plan
