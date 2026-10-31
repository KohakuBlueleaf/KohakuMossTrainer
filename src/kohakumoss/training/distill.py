"""Distillation towards a frozen teacher DiT: the teacher's velocity at the student's
inputs (same z_sigma, sigma, text) is a second target beside the data target.

Knobs: TEACHER ("" off, "weights" the run's own DiT file, else a DiT path),
DISTILL_WEIGHT, HARD_WEIGHT (the data loss), DISTILL_WEIGHTING (a WEIGHTINGS spec over
sigma, default the run's WEIGHTING), TEACHER_PRECISION (the teacher's base precision:
"auto" = as stored, or a quantized one such as "mxfp8"), TEACHER_STREAM ("compute" or
"fp32" residual streams). The teacher computes in bf16. See
docs/concepts/training.md#distillation."""

from pathlib import Path

import torch
from torch import Tensor

from kohakumoss.inference.loading import load_dit
from kohakumoss.logging.log import get_logger, hl, timed
from kohakumoss.memory.profiles import resolve as resolve_memory
from kohakumoss.model.io.load import read_fp16_fix
from kohakumoss.offload.accel import default_type
from kohakumoss.training.weights import component
from kohakumoss.weights import DIT

log = get_logger(__name__)
STREAMS = {"compute": "none", "fp32": "fp32_residual"}
TEACHER_DTYPE = torch.bfloat16
TEACHER_PROFILE = "speed"
OWN_WEIGHTS = "weights"


class Teacher:
    """`teacher(prepared)` -> the frozen model's velocity at the prepared inputs."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @torch.no_grad()
    def __call__(self, p) -> Tensor:
        return self.model(p.z, p.text, p.sigma, p.seq)


def teacher_path(k: dict) -> str:
    """The teacher DiT path of TEACHER ("" when off)."""
    choice = k.get("TEACHER", "")
    if choice == OWN_WEIGHTS:
        return str(component(k, DIT))
    return choice


def distill_problems(k: dict) -> list[str]:
    """TEACHER set: a positive DISTILL_WEIGHT, an existing DiT, a known stream mode
    (fp32 streams need a teacher file without a weight fold), a TEACHER_PRECISION this
    device type runs (the teacher's inference plan resolves), a non-negative
    HARD_WEIGHT."""
    p = []
    if float(k.get("HARD_WEIGHT", 1.0)) < 0:
        p.append("HARD_WEIGHT must be >= 0")
    if not k.get("TEACHER"):
        if float(k.get("DISTILL_WEIGHT", 0.0)) > 0:
            p.append("DISTILL_WEIGHT > 0 needs a TEACHER")
        return p
    if float(k.get("DISTILL_WEIGHT", 0.0)) <= 0:
        p.append("TEACHER needs DISTILL_WEIGHT > 0")
    stream = k.get("TEACHER_STREAM", "compute")
    if stream not in STREAMS:
        p.append(f"TEACHER_STREAM={stream!r} not in {tuple(STREAMS)}")
    try:
        path = teacher_path(k)
    except FileNotFoundError as err:
        return p + [f"TEACHER={k['TEACHER']!r}: {err}"]
    if not Path(path).exists():
        p.append(f"TEACHER={path!r} does not exist")
    elif stream == "fp32" and read_fp16_fix(path) is not None:
        p.append(
            "TEACHER_STREAM='fp32' needs a teacher file without an fp16-fix fold "
            "(the original weights)"
        )
    if k.get("PARALLEL", "none") not in ("none", "ddp"):
        p.append("TEACHER runs with PARALLEL 'none' or 'ddp'")
    precision = k.get("TEACHER_PRECISION", "auto")
    try:
        resolve_memory(
            TEACHER_PROFILE,
            {"base_precision": precision},
            default_type(),
            inference=True,
        )
    except ValueError as err:
        p.append(f"TEACHER_PRECISION={precision!r}: {err}")
    return p


def build_teacher(k: dict, device: torch.device) -> Teacher | None:
    """The frozen teacher on `device` (None without TEACHER): resident, bf16 compute,
    its base at TEACHER_PRECISION, residual streams per TEACHER_STREAM."""
    if not k.get("TEACHER"):
        return None
    path = teacher_path(k)
    precision = k.get("TEACHER_PRECISION", "auto")
    stream = k.get("TEACHER_STREAM", "compute")
    what = f"teacher {hl(path, 'path')} ({precision} base, {stream} stream)"
    with timed(log, what):
        loaded = load_dit(
            Path(path),
            TEACHER_DTYPE,
            device,
            (),
            "merge",
            STREAMS[stream],
            {},
            k["KERNELS"],
            TEACHER_PROFILE,
            {"base_precision": precision},
        )
    model = loaded.backbone.eval().requires_grad_(False)
    return Teacher(model)
