"""OUTLIER_DECAY on a real full finetune (device-stepped AdamW, every weight on the
card: a card of at least FULL_FT_BYTES), 3 steps, the plain and the deferred step: the
step logs the elements over the cap and their largest |p| / (cap r0), which falls every
step at a rate that shrinks the excess by lr x rate per step."""

import json
import math
from pathlib import Path

import pytest
import torch

from kohakumoss.training.entry import run
from tests._util.negative import must_fail
from tests.integration.conftest import needs_precision, real_knobs

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]

STEPS = 3
FULL_FT_BYTES = 90e9
LR = 1e-5
DECAY = {"rate": 2000.0, "cap": 16.0}
FULL = {"ALGO": "full", "LR": LR, "WEIGHT_DECAY": 0.0, "EXPORT_FORMATS": ()}
ARMS = {
    "plain": {"MEMORY_PROFILE": "ckpt-resident"},
    "deferred": {"MEMORY_PROFILE": "ckpt-resident", "STEP_SYNC": "deferred"},
}


def _rows(out: Path) -> list[dict]:
    text = (out / "run" / "metrics.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def check_decaying(rows: list[dict]) -> None:
    """Every step logs a positive count over the cap and a finite largest ratio above 1,
    and the ratio falls from step to step."""
    assert [r["step"] for r in rows] == list(range(1, STEPS + 1)), rows
    for r in rows:
        assert r["outlier_over"] > 0 and math.isfinite(r["outlier_max"]), r
        assert r["outlier_max"] > 1.0, r
    ratios = [r["outlier_max"] for r in rows]
    assert all(b < a for a, b in zip(ratios, ratios[1:])), ratios


@pytest.mark.parametrize("arm", list(ARMS))
def test_outlier_decay_full_finetune(arm, weights_root, dataset_slice, tmp_path):
    needs_precision("bf16")
    total = torch.cuda.get_device_properties(0).total_memory
    if total < FULL_FT_BYTES:
        pytest.skip(
            f"a device-stepped full finetune needs {FULL_FT_BYTES / 1e9:.0f} GB"
        )
    k = real_knobs(
        weights_root,
        dataset_slice,
        tmp_path,
        PRECISION="bf16",
        MAX_TOKENS=2048,
        OUTLIER_DECAY=DECAY,
        **FULL,
        **ARMS[arm],
    )
    assert run(k).step == STEPS
    rows = _rows(tmp_path)
    print(
        f"{arm}: "
        + ", ".join(f"{r['outlier_over']:.0f} / {r['outlier_max']:.3f}" for r in rows)
    )
    check_decaying(rows)
    flat = [{**r, "outlier_max": rows[0]["outlier_max"]} for r in rows]
    must_fail(lambda: check_decaying(flat), "a ratio that does not fall")
