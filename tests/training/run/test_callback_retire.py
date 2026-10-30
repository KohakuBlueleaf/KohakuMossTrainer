"""A callback whose on_step_end returns True has finished and leaves the run: the runner
never calls it again, every-step and periodic alike."""

import pytest

from kohakumoss.training.callbacks.runner import CallbackRunner
from kohakumoss.training.state import TrainState
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(5)]


class _Once:
    def __init__(self, every: int, done_at: int):
        self.every = every
        self.done_at = done_at
        self.calls = []

    def on_step_end(self, state, out):
        self.calls.append(state.step)
        return state.step >= self.done_at


class _Keep:
    every = 1

    def __init__(self):
        self.calls = 0

    def on_step_end(self, state, out):
        self.calls += 1


def _run(runner_cls) -> None:
    once, periodic, keep = _Once(1, 3), _Once(4, 8), _Keep()
    runner = runner_cls([once, periodic, keep], last=20)
    for step in range(1, 21):
        runner.run(TrainState(step=step), None)
    assert once.calls == [1, 2, 3]
    assert periodic.calls == [4, 8]
    assert keep.calls == 20


def test_finished_callbacks_leave():
    _run(CallbackRunner)


def test_finished_callbacks_leave_negative():
    class Keeps(CallbackRunner):
        def _drop(self, done):
            return None

    must_fail(lambda: _run(Keeps), "finished callbacks still called")
