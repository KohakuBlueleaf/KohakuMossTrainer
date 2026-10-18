"""The adapter autotuned-op prewarm on a restart: identical layers are one op, and the
second process against the same kernel cache times nothing."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests._util.negative import must_fail

ROOT = Path(__file__).resolve().parents[3]
CHILD = "tests.training.prewarm._adapter_prewarm_twice"


def _child(cache: Path, *flags: str) -> dict:
    env = {**os.environ, "KOHAKUMOSS_KERNEL_CACHE": str(cache)}
    out = subprocess.run(
        [sys.executable, "-m", CHILD, *flags],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-6000:]
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    return json.loads(lines[-1])


def check_restart(first: dict, second: dict) -> None:
    assert first["distinct"] < first["ops"], first
    assert first["measured"] > 0 and first["timed"] > 0, first
    assert second["measured"] == 0 and second["timed"] == 0, second


@pytest.mark.cuda
@pytest.mark.budget(60)  # two child processes
def test_adapter_restart_times_nothing(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    check_restart(_child(tmp_path), _child(tmp_path))


@pytest.mark.cuda
@pytest.mark.budget(60)  # two child processes
def test_negative_adapter_restart_without_dedupe_or_table(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    first, second = _child(tmp_path, "fresh"), _child(tmp_path, "fresh")
    must_fail(lambda: check_restart(first, second), "every op and bucket timed")
