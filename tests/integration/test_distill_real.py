"""Distillation on real training runs: LoRA (zero-initialised up factors) with the run's
own DiT as TEACHER, 3 steps. Every loss is finite and `loss_distill` is logged each
step. At step 1 the student is the teacher's function, so `loss_distill` is only the
two paths' numeric difference: below DISTILL_FLOOR x `loss_task` for a bf16 teacher;
an mxfp8 teacher (block-scaled e4m3, quantized activations) logs its own step-1 gap,
which must stay below MX_CEILING x `loss_task`."""

import json
import math
from pathlib import Path

import pytest

from kohakumoss.training.entry import run
from kohakumoss_kernels.hardware.arch import arch_of
from tests._util.negative import must_fail
from tests.integration.conftest import needs_precision, real_knobs

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]

STEPS = 3
TOKENS = 2048
DISTILL_FLOOR = 1e-2
MX_CEILING = 0.2
LORA = {"ALGO": "lora", "DIM": 8, "ALPHA": 8.0, "LR": 5e-4}


def _rows(out: Path) -> list[dict]:
    text = (out / "run" / "metrics.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def check_distilled(rows: list[dict], ceiling: float) -> None:
    """Every step logged with finite total, data and teacher losses; the step-1 teacher
    loss below `ceiling` x the step-1 data loss."""
    assert [r["step"] for r in rows] == list(range(1, STEPS + 1)), rows
    for r in rows:
        for key in ("loss", "loss_task", "loss_distill"):
            assert math.isfinite(r[key]), (key, r)
    first = rows[0]
    assert first["loss_distill"] < ceiling * first["loss_task"], first


def _knobs(weights_root, dataset, out, precision: str) -> dict:
    return real_knobs(
        weights_root,
        dataset,
        out,
        MEMORY_PROFILE="low",
        MAX_TOKENS=TOKENS,
        PRECISION="bf16",
        TEACHER="weights",
        DISTILL_WEIGHT=1.0,
        TEACHER_PRECISION=precision,
        **LORA,
    )


@pytest.mark.parametrize(
    ("precision", "ceiling"), [("auto", DISTILL_FLOOR), ("mxfp8", MX_CEILING)]
)
def test_teacher_matches_the_untrained_student(
    precision, ceiling, weights_root, dataset_slice, tmp_path
):
    needs_precision("bf16")
    if precision == "mxfp8" and not arch_of("cuda").has_fp8:
        pytest.skip("mxfp8 needs fp8 GEMMs (sm89+)")
    state = run(_knobs(weights_root, dataset_slice, tmp_path, precision))
    assert state.step == STEPS
    rows = _rows(tmp_path)
    print(
        f"teacher {precision}: step-1 loss_distill {rows[0]['loss_distill']:.3e} "
        f"loss_task {rows[0]['loss_task']:.3e}"
    )
    check_distilled(rows, ceiling)
    far = [{**r, "loss_distill": r["loss_task"]} for r in rows]
    must_fail(lambda: check_distilled(far, ceiling), "a teacher of another function")
