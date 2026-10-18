"""Multi-GPU knob rules: every multi-device layout is rejected on Windows."""

import sys

import pytest

from kohakumoss.training.knobs import validate
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu


def test_multi_gpu_rejected_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    for kind in ("ddp", "fsdp", "pipeline", "threaded"):
        problems = validate(runnable(PARALLEL=kind, DEVICES=(0, 1), MICROBATCHES=2))
        assert any("Linux-only" in p for p in problems), kind
    several = validate(runnable(DEVICES=(0, 1)))
    assert any("Linux-only" in p for p in several), several
    assert validate(runnable()) == []
    monkeypatch.setattr(sys, "platform", "linux")
    for kind in ("ddp", "fsdp", "threaded"):
        problems = validate(runnable(PARALLEL=kind, DEVICES=(0, 1)))
        assert not any("Linux-only" in p for p in problems), kind
