"""TimedCheckpoint: every rank takes the same save decision (the save is collective),
checked only on its cadence."""

import pytest

from kohakumoss.training.callbacks import CallbackRunner
from kohakumoss.training.callbacks.runner import TimedCheckpoint
from kohakumoss.training.state import TrainState
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


class _Ckpt:
    def __init__(self):
        self.steps = []

    def save(self, step, parts):
        self.steps.append(step)


def _ranks(decisions):
    """Two ranks whose clocks disagree; `decisions(due)` is the agreement rule."""
    saved = []
    for clock_due in (True, False):
        ckpt = _Ckpt()
        cb = TimedCheckpoint(
            0.0, ckpt, lambda s: {}, every=1, agree=decisions(clock_due)
        )
        cb.seconds = -1.0 if clock_due else 1e9
        cb.on_step_end(TrainState(step=1), None)
        saved.append(ckpt.steps)
    return saved


def _agree(saved) -> None:
    assert saved[0] == saved[1], saved


def test_ranks_agree_on_saving():
    rank0_due = True
    _agree(_ranks(lambda own: (lambda due: rank0_due)))
    must_fail(lambda: _agree(_ranks(lambda own: (lambda due: due))), "local clocks")


def test_checked_on_cadence_only():
    ckpt = _Ckpt()
    cb = TimedCheckpoint(0.0, ckpt, lambda s: {}, every=5)
    runner = CallbackRunner([cb])
    for step in range(1, 11):
        runner.run(TrainState(step=step), None)
    assert ckpt.steps == [5, 10]
