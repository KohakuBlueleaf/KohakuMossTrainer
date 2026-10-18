"""Training progress, model-free: the callback edges (LOG_EVERY plus the run's first
and last step), one bar over STEPS with the postfix keys and an EMA-smoothed loss,
closed after the last step; the bar only on a TTY or FORCE_COLOR; METRICS_EVERY
checked. The cadence of a real run is checked in tests/integration/test_train_real.py.
"""

import io

import pytest

from kohakumoss.logging.log import interactive
from kohakumoss.training.callbacks.progress import POSTFIX, ProgressBar
from kohakumoss.training.callbacks.runner import CallbackRunner
from kohakumoss.training.knobs import validate
from kohakumoss.training.state import TrainState
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu


class FakeBar:
    made: list["FakeBar"] = []

    def __init__(self, **kw):
        self.kw, self.n, self.postfix, self.closed = kw, kw["initial"], {}, False
        FakeBar.made.append(self)

    def set_postfix(self, post, refresh=True):
        self.postfix = dict(post)

    def update(self, k=1):
        self.n += k

    def close(self):
        self.closed = True


class Rec:
    def __init__(self, every, edges, seen):
        self.every, self.edges, self.seen = every, edges, seen

    def on_step_end(self, state, out):
        self.seen.append(state.step)


def _fired(start: int, every: int, edges: bool, last: int = 7) -> list[int]:
    seen: list[int] = []
    runner = CallbackRunner([Rec(every, edges, seen)], last=last)
    runner.reset(start)
    for step in range(start + 1, last + 1):
        runner.run(TrainState(step=step), None)
    return seen


def _equal(a, b) -> None:
    assert a == b, (a, b)


def test_runner_fires_on_the_edges():
    assert _fired(0, 3, True) == [1, 3, 6, 7]
    assert _fired(4, 3, True) == [5, 6, 7]
    assert _fired(0, 3, False) == [3, 6]
    must_fail(lambda: _equal(_fired(0, 1, True), [1, 3, 6, 7]), "LOG_EVERY ignored")


def test_bar_postfix_and_ema():
    FakeBar.made = []
    ticks = iter(range(100))
    pb = ProgressBar(4, "cpu", clock=lambda: float(next(ticks)), make=FakeBar)
    for step, loss in enumerate([1.0, 2.0, 2.0, 2.0], start=1):
        pb.on_step_end(
            TrainState(step=step, loss=loss, lr=1e-4, samples=2 * step), None
        )
    fake = FakeBar.made[0]
    assert fake.kw["total"] == 4 and fake.kw["initial"] == 0 and fake.n == 4
    assert tuple(fake.postfix) == POSTFIX and fake.closed
    assert fake.postfix["samples/s"] == "2.0" and fake.postfix["s/it"] == "1.00"
    ema = 0.9 * (0.9 * (0.9 * 1.0 + 0.1 * 2.0) + 0.1 * 2.0) + 0.1 * 2.0
    assert fake.postfix["loss"] == f"{ema:.4f}"
    must_fail(lambda: _equal(fake.postfix["loss"], "2.0000"), "the raw last loss")


class Tty(io.StringIO):
    def isatty(self):
        return True


def test_bar_only_on_a_terminal(monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert interactive(Tty()) and not interactive(io.StringIO())
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert interactive(io.StringIO())
    monkeypatch.setenv("FORCE_COLOR", "0")
    must_fail(lambda: _equal(interactive(io.StringIO()), True), "a pipe with a bar")


def _metric_problems(value) -> list[str]:
    return [p for p in validate(runnable(METRICS_EVERY=value)) if "METRICS_EVERY" in p]


def test_metrics_every_is_checked():
    for bad in (0, -1, 2.5, True):
        assert _metric_problems(bad), bad
    for good in (None, 1, 4):
        assert _metric_problems(good) == [], good
    must_fail(lambda: _equal(_metric_problems(0), []), "a zero cadence accepted")
