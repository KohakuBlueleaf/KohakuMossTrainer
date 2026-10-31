"""TEACHER / DISTILL_WEIGHT / HARD_WEIGHT / TEACHER_STREAM knob checks (pure functions)."""

import pytest

from kohakumoss.training.distill import distill_problems

pytestmark = [pytest.mark.cpu, pytest.mark.budget(5)]


def test_off_is_valid():
    assert distill_problems({}) == []
    assert distill_problems({"TEACHER": "", "DISTILL_WEIGHT": 0.0}) == []


def test_weight_without_teacher():
    assert distill_problems({"DISTILL_WEIGHT": 0.5})


def test_teacher_needs_weight_and_existing_path(tmp_path):
    ok = {"TEACHER": str(tmp_path), "DISTILL_WEIGHT": 1.0}
    assert distill_problems(ok) == []
    assert distill_problems({**ok, "DISTILL_WEIGHT": 0.0})
    assert distill_problems({**ok, "TEACHER": str(tmp_path / "missing")})
    assert distill_problems({**ok, "TEACHER_STREAM": "fp64"})
    assert distill_problems({**ok, "PARALLEL": "fsdp"})
    assert distill_problems({**ok, "HARD_WEIGHT": -1.0})
    assert distill_problems({**ok, "TEACHER_PRECISION": "fp5"})
