"""Memory profiles: named presets over MemoryPlan knobs, resolved once per run.

See docs/internals/offload.md for what each knob trades.
"""

from dataclasses import fields, replace
from itertools import product

import torch

from kohakumoss.quant.compute import COMPUTE as FP8_COMPUTE
from kohakumoss.quant.compute import DGRAD as FP8_DGRAD
from kohakumoss.quant.compute import FP8_RECIPES
from kohakumoss.types.plans import MemoryPlan
from kohakumoss_common.log import get_logger
from kohakumoss_common.typecheck import describe, mismatches
from kohakumoss_kernels.hardware.arch import arch_of

CHECKPOINT_MODES = ("none", "block", "sub-block", "selective")
STREAM_MODES = ("off", "ring", "swap")
# a streamed base by (stream asked, frozen base): ring streams a frozen base, swap a trainable one
STREAM_FOR = {
    "ring": {True: "ring", False: "swap"},
    "swap": {True: "ring", False: "swap"},
}
_LOGGED: set[tuple[str, str]] = set()
log = get_logger(__name__)
BASE_PRECISIONS = ("auto", "compute", "fp8", "int8", "quant", *FP8_COMPUTE)
QUANTIZED_BASES = ("fp8", "int8", "quant")
KNOB_FIELDS = {"compile": "COMPILE", "kernels": "KERNELS"}
OPTIM_STATES = ("gpu", "stream", "cpu")
COMPILE_MODES = ("off", "block", "model")

PROFILES: dict[str, dict] = {
    "speed": dict(
        checkpoint="none",
        act_offload=False,
        base_precision="auto",
        stream="off",
        optim_state="gpu",
        fused_step=False,
    ),
    "balanced": dict(
        checkpoint="block",
        act_offload=False,
        base_precision="auto",
        stream="ring",
        stream_resident=16,
        optim_state="gpu",
        fused_step=False,
    ),
    "low": dict(
        checkpoint="block",
        act_offload=True,
        base_precision="auto",
        stream="ring",
        stream_resident=4,
        optim_state="stream",
        fused_step=True,
    ),
    "minimal": dict(
        checkpoint="block",
        act_offload=True,
        base_precision="auto",
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


def _fp8_device_problems(plan: MemoryPlan, device) -> list[str]:
    if plan.base_precision not in FP8_COMPUTE or device is None:
        return []
    kind = torch.device(device).type
    if kind == "cuda":
        return []
    return [f"base_precision {plan.base_precision!r} needs a CUDA device, not {kind}"]


def resolve_base_precision(value: str, device: torch.device | str) -> str:
    """`quant` picks fp8 where the arch has fp8 GEMMs, else int8; `auto` stays (the
    checkpoint's stored format, applied by `memory.apply.quantize_base`)."""
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
    """Every problem with `plan`, as messages (empty = valid); a field of the wrong type
    is reported alone, ahead of the checks that read its value."""
    problems = [
        f"{name}={value!r} is not {describe(annotation)}"
        for name, value, annotation in mismatches(plan)
    ]
    if problems:
        return problems
    if (
        device is not None
        and plan.base_precision in ("fp8", *FP8_COMPUTE)
        and not arch_of(device).has_fp8
    ):
        problems.append(
            f"base_precision {plan.base_precision!r} needs fp8 GEMMs (sm89+), not "
            f"{arch_of(device).name}"
        )
    if plan.fp8_recipe not in FP8_RECIPES:
        problems.append(f"fp8_recipe {plan.fp8_recipe!r} not in {FP8_RECIPES}")
    if plan.fp8_dgrad not in FP8_DGRAD:
        problems.append(f"fp8_dgrad {plan.fp8_dgrad!r} not in {FP8_DGRAD}")
    mode, frac = checkpoint_mode(plan.checkpoint)
    if mode not in CHECKPOINT_MODES:
        problems.append(f"checkpoint {plan.checkpoint!r} not in {CHECKPOINT_MODES}")
    if not 0.0 <= frac <= 1.0:
        problems.append(f"checkpoint fraction {frac} outside [0, 1]")
    if plan.stream not in STREAM_MODES:
        problems.append(f"stream {plan.stream!r} not in {STREAM_MODES}")
    if plan.stream != "off" and plan.stream_resident < 2:
        problems.append("weight streaming needs stream_resident >= 2")
    if plan.act_offload and plan.act_offload_inflight < 1:
        problems.append("activation offload needs act_offload_inflight >= 1")
    if plan.base_precision not in BASE_PRECISIONS:
        problems.append(
            f"base_precision {plan.base_precision!r} not in {BASE_PRECISIONS}"
        )
    if plan.base_precision in QUANTIZED_BASES and trainable_base:
        problems.append("a quantized base cannot be trained (full FT)")
    if plan.base_precision in FP8_COMPUTE and trainable_base and plan.stream != "off":
        problems.append(
            f"base_precision {plan.base_precision!r} full FT keeps the weights resident "
            "(fp8 copies requantized after each step): stream='off'"
        )
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
    return problems + _fp8_device_problems(plan, device)


def _log_stream_once(asked: str, used: str, frozen: bool) -> None:
    """One INFO line per process for each streamed-base mode the base turned into."""
    if (asked, used) in _LOGGED:
        return
    _LOGGED.add((asked, used))
    base = "a frozen base" if frozen else "a trainable base (full FT)"
    log.info(f"stream={asked!r} on {base} streams as {used!r}")


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
    checkpointing. A streamed base takes the mode its base needs (`STREAM_FOR`: ring
    for a frozen base, swap for full FT; an explicit other mode is logged once).
    `profile` may use the old ring spelling (`profile_name`)."""
    profile = profile_name(profile)
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
        values["base_precision"] = "auto"
    frozen = not trainable_base or inference
    stream = values.get("stream")
    if stream in STREAM_FOR and STREAM_FOR[stream][frozen] != stream:
        values["stream"] = STREAM_FOR[stream][frozen]
        if "stream" in overrides:
            _log_stream_once(stream, values["stream"], frozen)
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
    plan = replace(MemoryPlan(), **values)
    plan = replace(
        plan, base_precision=resolve_base_precision(plan.base_precision, device)
    )
    problems = validate(plan, trainable_base and not inference, inference, device)
    if problems:
        raise ValueError("invalid memory plan: " + "; ".join(problems))
    return plan


CATALOG_AXES: tuple[dict[str, dict], ...] = (
    {
        "nockpt": dict(checkpoint="none"),
        "ckpt": dict(checkpoint="block"),
        "subckpt": dict(checkpoint="sub-block"),
    },
    {
        "resident": dict(stream="off"),
        "ring16": dict(stream="ring", stream_resident=16),
        "ring8": dict(stream="ring", stream_resident=8),
        "ring4": dict(stream="ring", stream_resident=4),
        "ring2": dict(stream="ring", stream_resident=2),
    },
    {"": dict(act_offload=False), "actoff": dict(act_offload=True)},
    {
        "": dict(optim_state="gpu", fused_step=False),
        "optfused": dict(optim_state="gpu", fused_step=True),
        "optstream": dict(optim_state="stream", fused_step=True),
        "optcpu": dict(optim_state="cpu", fused_step=False),
    },
    {"": dict(grad_offload=False), "gradoff": dict(grad_offload=True)},
)


RING_ALIASES = {f"stream{n}": f"ring{n}" for n in (16, 8, 4, 2)}


def profile_name(name):
    """A profile name in its primary spelling: the weight-ring parts `stream16` ..
    `stream2` read as `ring16` .. `ring2`; any other value is returned as is."""
    if not isinstance(name, str):
        return name
    return "-".join(RING_ALIASES.get(part, part) for part in name.split("-"))


def catalog() -> dict[str, dict]:
    """Every combination of the placement axes (checkpoint x weights x activations x
    optimizer x gradients) that validates for a frozen-base run, named by its parts;
    precision comes from the precision knobs. See docs/concepts/profiles.md."""
    out: dict[str, dict] = {}
    for combo in product(*(axis.items() for axis in CATALOG_AXES)):
        name = "-".join(part for part, _ in combo if part)
        values = dict(base_precision="auto", pinned=True)
        for _, knobs in combo:
            values.update(knobs)
        if not validate(replace(MemoryPlan(), **values)):
            out[name] = values
    return out


CATALOG = catalog()
PROFILES.update(CATALOG)
PROFILES["manual"] = {}


def catalog_name(profile: str) -> str | None:
    """The catalog entry whose plan equals `profile`'s (a short name), or None."""
    plan = replace(MemoryPlan(), **PROFILES[profile_name(profile)])
    for name, values in CATALOG.items():
        if replace(MemoryPlan(), **values) == plan:
            return name
    return None
