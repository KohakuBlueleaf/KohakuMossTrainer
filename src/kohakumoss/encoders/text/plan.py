"""TextEncoderPlan from a knob dict, and every problem with one (training and inference
share the rules)."""

from dataclasses import fields, replace
from pathlib import Path

import torch

from kohakumoss.quant.compute import COMPUTE, FP8_RECIPES
from kohakumoss.types.plans import TextEncoderPlan
from kohakumoss_kernels import ops as kernel_ops
from kohakumoss_kernels.dtypes import BY_NAME
from kohakumoss_kernels.hardware.arch import arch_of

MEMORY = ("resident", "offload", "ring", "cpu", "auto")
MEMORY_ALIASES = {"stream": "ring"}
PRECISION = ("auto", "compute", "int8", "fp8", *COMPUTE, "w4a8")
STORED_ONLY = ("w4a8",)
HOST_PRECISIONS = ("compute", "int8", "fp8", "w4a8")
W4A8 = ("auto", "int8", "packed")
LAYOUT = ("single", "padded", "packed")
FORWARD = ("kernels", "hf")
DTYPES = {name: BY_NAME[name] for name in ("bf16", "fp16")}
AHEAD_DEVICE = ("host", "device")
HOST_FORMS = ("auto", "copy", "file")


def text_plan(
    spec: dict | TextEncoderPlan | None = None, **overrides
) -> TextEncoderPlan:
    """A TextEncoderPlan from a plan, a dict of its fields, or defaults, plus overrides,
    with `memory` in its primary spelling (`MEMORY_ALIASES`: "stream" is "ring");
    raises listing every unknown field."""
    if isinstance(spec, TextEncoderPlan):
        base, values = spec, {}
    else:
        base, values = TextEncoderPlan(), dict(spec or {})
    values.update(overrides)
    known = {f.name for f in fields(TextEncoderPlan)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(
            f"unknown text-encoder plan fields {unknown}; valid {sorted(known)}"
        )
    plan = replace(base, **values)
    return replace(plan, memory=MEMORY_ALIASES.get(plan.memory, plan.memory))


def resolve_precision(
    plan: TextEncoderPlan, stored: str, device: torch.device | str
) -> TextEncoderPlan:
    """`precision="auto"`: the format the weights store (`stored`: `compute` for a
    16-bit checkpoint); on the host (memory 'cpu') a stored format the host cannot run
    (`HOST_PRECISIONS`: fp8 compute GEMMs need CUDA) is `compute`; ValueError for a
    stored-only precision (`w4a8`) the weights do not store."""
    if plan.precision in STORED_ONLY and stored != plan.precision:
        raise ValueError(
            f"text-encoder precision {plan.precision!r} is a stored format; these "
            f"weights store {stored!r}: use 'auto' or a precision KohakuMoss quantizes "
            "to"
        )
    if plan.precision != "auto":
        return plan
    on_host = plan.memory == "cpu" or torch.device(device).type == "cpu"
    runs = not on_host or stored in HOST_PRECISIONS
    return replace(plan, precision=stored if runs else "compute")


def _has_bf16(device: torch.device | str) -> bool:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return True
    return arch_of(device).has_bf16


def resolve_dtype(
    plan: TextEncoderPlan, *devices: torch.device | str
) -> TextEncoderPlan:
    """`dtype="auto"`: bf16 where every one of `devices` has it (the device the
    embeddings are for, whatever the placement, and the device the encoder computes
    on; the host and a CUDA device this process cannot query have it), else fp16."""
    if plan.dtype != "auto":
        return plan
    has_bf16 = all(_has_bf16(d) for d in devices)
    return replace(plan, dtype="bf16" if has_bf16 else "fp16")


def plan_problems(
    plan: TextEncoderPlan,
    device: torch.device | str | None = None,
    arch=arch_of,
) -> list[str]:
    """Every problem with `plan` (empty = valid). `arch(device)` gives the CUDA arch
    (default arch_of)."""
    p = []
    for name, value, opts in (
        ("memory", plan.memory, MEMORY),
        ("precision", plan.precision, PRECISION),
        ("fp8_recipe", plan.fp8_recipe, FP8_RECIPES),
        ("layout", plan.layout, LAYOUT),
        ("forward", plan.forward, FORWARD),
        ("dtype", plan.dtype, (*DTYPES, "auto")),
        ("ahead_device", plan.ahead_device, AHEAD_DEVICE),
        ("w4a8", plan.w4a8, W4A8),
        ("host", plan.host, HOST_FORMS),
    ):
        if value not in opts:
            p.append(f"text-encoder {name}={value!r} not in {opts}")
    if plan.forward == "hf" and plan.layout == "packed":
        p.append(
            "text-encoder forward='hf' runs layout 'single' or 'padded'; 'packed' needs "
            "forward='kernels'"
        )
    if plan.batch < 1 or plan.ahead < 1 or plan.max_tokens < 0:
        p.append("text-encoder batch >= 1, ahead >= 1 and max_tokens >= 0")
    if plan.memory == "ring" and plan.stream_resident < 2:
        p.append("text-encoder ring needs stream_resident >= 2")
    if plan.quant_group and plan.precision not in ("int8", "fp8"):
        p.append(
            "text-encoder quant_group is a weight-only layout: precision int8 / fp8 "
            "(fp8_compute / mxfp8 scale by fp8_recipe)"
        )
    if plan.quant_group < 0 or (plan.quant_group and plan.quant_group % 16):
        p.append("text-encoder quant_group is 0 (per channel) or a multiple of 16")
    if plan.stored and (plan.precision == "compute" or plan.quant_group):
        p.append(
            "text-encoder stored (a pre-quantized file) needs precision auto / int8 / "
            "fp8 / fp8_compute / mxfp8 (the file's) and quant_group 0"
        )
    if plan.stored and not Path(plan.stored).is_file():
        p.append(f"text-encoder stored file {plan.stored} does not exist")
    bad_ops = sorted(set(plan.kernels) - set(kernel_ops.TEXT) - {"rms_norm"})
    if bad_ops:
        p.append(f"text-encoder kernels has unknown ops {bad_ops}")
    if plan.memory == "cpu" and plan.precision not in ("auto", *HOST_PRECISIONS):
        p.append(
            f"text-encoder memory='cpu' runs precision auto / {' / '.join(HOST_PRECISIONS)}"
            " (fp8 compute GEMMs need CUDA)"
        )
    if plan.int8_compute and plan.precision not in ("auto", "int8", "w4a8"):
        p.append(
            f"text-encoder int8_compute runs int8 linears; precision {plan.precision!r} "
            "has none: use 'auto' on an int8 / W4A8 file, or 'int8'"
        )
    if plan.int8_compute and plan.quant_group:
        p.append("text-encoder int8_compute runs per-channel int8: quant_group 0")
    if plan.int8_compute and plan.memory == "cpu":
        p.append("text-encoder int8_compute needs a CUDA device, not memory 'cpu'")
    if plan.int8_compute and device is not None and torch.device(device).type != "cuda":
        p.append(
            f"text-encoder int8_compute needs a CUDA device, not "
            f"{torch.device(device).type}"
        )
    if plan.precision in COMPUTE and plan.memory != "cpu" and device is not None:
        kind = torch.device(device).type
        if kind != "cuda":
            p.append(f"text-encoder {plan.precision} needs a CUDA device, not {kind}")
    if device is not None and torch.device(device).type == "cuda":
        info = arch(device)
        if plan.precision in COMPUTE and not info.has_fp8:
            p.append(
                f"text-encoder {plan.precision} needs fp8 GEMMs (sm89+), not "
                f"{info.name}"
            )
        if plan.dtype == "bf16" and not info.has_bf16 and plan.memory != "cpu":
            p.append(
                f"text-encoder dtype 'bf16' on {info.name} (no bf16): use fp16 or auto"
            )
    return p


def check_plan(plan: TextEncoderPlan, **kw) -> TextEncoderPlan:
    problems = plan_problems(plan, **kw)
    if problems:
        raise ValueError("invalid text-encoder plan: " + "; ".join(problems))
    return plan
