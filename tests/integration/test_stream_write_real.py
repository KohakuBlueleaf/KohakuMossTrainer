"""STREAM_LOSS write term on real training runs: LoRA, 3 steps, bf16, the hinge active
from step 1 (target 0.5 against base write shares near 0.9). The same run with block
checkpointing, without it, and with block checkpointing under COMPILE "block" logs the
same write hinges and task losses at every step (the write taps recompute what their
forward saved, so the gradients after step 1 agree); a run at target 0.7 does not."""

import json
import math
from pathlib import Path

import pytest

from kohakumoss.training.entry import run
from tests._util.negative import must_fail
from tests.integration.conftest import needs_precision, real_knobs

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]

STEPS = 3
TOKENS = 2048
REL = 5e-3
LORA = {"ALGO": "lora", "DIM": 8, "ALPHA": 8.0, "LR": 5e-4}
WRITE = {"write": {"weight": 1.0, "target": 0.5}}
ARMS = {
    "ckpt": {"MEMORY_PROFILE": "ckpt-resident"},
    "nockpt": {"MEMORY_PROFILE": "nockpt-resident"},
    "ckpt_compiled": {"MEMORY_PROFILE": "ckpt-resident", "COMPILE": "block"},
}


def _rows(out: Path) -> list[dict]:
    text = (out / "run" / "metrics.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _run(weights_root, dataset, out: Path, stream: dict, **arm) -> list[dict]:
    k = real_knobs(
        weights_root,
        dataset,
        out,
        PRECISION="bf16",
        MAX_TOKENS=TOKENS,
        STREAM_LOSS=stream,
        **LORA,
        **arm,
    )
    assert run(k).step == STEPS
    return _rows(out)


def check_written(rows: list[dict]) -> None:
    """Every step logs a finite positive write hinge, part shares in (0.5, 1] and the
    target 0.5."""
    assert [r["step"] for r in rows] == list(range(1, STEPS + 1)), rows
    for r in rows:
        assert math.isfinite(r["loss"]) and r["write_loss"] > 0, r
        for part in ("write_share_attn", "write_share_ffn"):
            assert 0.5 < r[part] <= 1.0, (part, r)
        assert r["write_target"] == pytest.approx(0.5)


def check_same(a: list[dict], b: list[dict]) -> None:
    """Two runs' per-step write hinges and task losses within REL of each other (the
    steps after the first carry the first steps' gradients)."""
    for x, y in zip(a, b):
        for key in ("write_loss", "loss_task"):
            assert x[key] == pytest.approx(y[key], rel=REL), (key, x, y)


def test_write_term_through_checkpoint_and_compile(
    weights_root, dataset_slice, tmp_path
):
    needs_precision("bf16")
    rows = {
        name: _run(weights_root, dataset_slice, tmp_path / name, WRITE, **arm)
        for name, arm in ARMS.items()
    }
    looser = {"write": {**WRITE["write"], "target": 0.7}}
    other = _run(
        weights_root, dataset_slice, tmp_path / "other", looser, **ARMS["ckpt"]
    )
    for name, r in rows.items():
        print(
            f"{name}: write_loss {[round(x['write_loss'], 5) for x in r]} "
            f"grad_norm {[round(x['grad_norm'], 4) for x in r]} "
            f"loss_task {[round(x['loss_task'], 5) for x in r]} "
            f"shares attn {r[0]['write_share_attn']:.3f} ffn {r[0]['write_share_ffn']:.3f}"
        )
    print(f"target 0.7: write_loss {[round(x['write_loss'], 5) for x in other]}")
    for r in rows.values():
        check_written(r)
        check_same(r, rows["ckpt"])
    must_fail(lambda: check_same(other, rows["ckpt"]), "a run at another target")
