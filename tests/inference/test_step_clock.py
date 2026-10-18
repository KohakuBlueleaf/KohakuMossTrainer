"""Sampler step diagnostics are read once per batch: `tick` only marks the step,
`summary` turns the marks into one row per step in the list the pipeline holds from
before the summary (model-free: the host clock)."""

import pytest
import torch

from kohakumoss.inference.steplog import StepClock
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def check_rows(clock: StepClock) -> None:
    clock.start()
    for i in range(3):
        clock.tick(i)
    assert clock.rows == [], "rows built before the batch ended"
    held = clock.rows
    out = clock.summary()
    assert [r["step"] for r in held] == [1, 2, 3], held
    assert all(r["s"] >= 0 for r in held) and out["mean_s"] >= 0


def _rebinding_summary(self) -> dict:
    """The broken variant: a new rows list, which the caller's earlier reference misses."""
    self.rows = self._rows()
    return {"mean_s": 0.0}


def test_rows_land_in_the_held_list(monkeypatch):
    check_rows(StepClock(torch.device("cpu"), 3))
    monkeypatch.setattr(StepClock, "summary", _rebinding_summary)
    must_fail(lambda: check_rows(StepClock(torch.device("cpu"), 3)), "rebound")
