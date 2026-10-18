"""The tuning windows, model-free: after `freeze` a missing key binds untimed and a
window times it; the between-step retune runs over the process's first N steps and then
seals the tuner; a sealed kernel-or-compile op queues nothing; only the first sample
grid's first sampler step is open."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.training.build.loop import PipelineImages
from kohakumoss.training.callbacks.kernels import RetuneKernels
from kohakumoss_kernels.backends.compile import choice, ops
from kohakumoss_kernels.backends.eager import ffn
from kohakumoss_kernels.hardware import memory
from kohakumoss_kernels.tuning import tuner, window
from tests._util.negative import must_fail
from tests.kernels.tuning.test_tuner_working import Allocator

pytestmark = pytest.mark.cpu
DEV = torch.device("cuda", 0)
FAST = {"ms": 1.0}
SLOW = {"ms": 2.0}


@pytest.fixture
def timed(monkeypatch, tmp_path):
    """A fake allocator, per-test tables and a timing that counts its measurements."""
    monkeypatch.setattr(memory, "MODULES", {"cuda": Allocator()})
    monkeypatch.setattr(memory, "_OPEN", {})
    tables = {}

    def table(kernel, device):
        return tables.setdefault(kernel, tuner.TuneTable(tmp_path / f"{kernel}.json"))

    measured = []

    def events(fn, device):
        measured.append(1)
        return fn()

    monkeypatch.setattr(tuner, "table", table)
    monkeypatch.setattr(tuner, "_CHOSEN", {})
    monkeypatch.setattr(tuner, "DEFERRED", set())
    monkeypatch.setattr(
        tuner, "TIMINGS", {"events": lambda fn, device: events(fn, device)}
    )
    tuner.set_mode("table")
    return SimpleNamespace(tables=tables, measured=measured)


def _select(key: str) -> dict:
    return tuner.select(
        "test_window", DEV, key, lambda: [SLOW, FAST], lambda c: lambda: c["ms"], 2
    )


def check_frozen(t, freeze) -> None:
    """Frozen: a missing key binds its first candidate, untimed and unstored; inside a
    window the next missing key is timed and stored; the freeze holds after it."""
    freeze()
    assert _select("late") == SLOW
    assert t.measured == [], f"{len(t.measured)} measurements after the freeze"
    assert (str(DEV), "test_window", "late") in tuner.DEFERRED
    assert "late" not in t.tables["test_window"].rows
    with window.tuning_window():
        assert _select("open") == FAST
    assert len(t.measured) == 2 and "open" in t.tables["test_window"].rows
    assert _select("after") == SLOW and len(t.measured) == 2


def test_frozen_keys_bind_untimed(timed):
    check_frozen(timed, window.freeze)


def test_negative_a_freeze_that_times(timed):
    must_fail(
        lambda: check_frozen(timed, lambda: None), "a select after the freeze timed"
    )


def test_window_close_and_seal():
    window.freeze()
    with window.tuning_window() as close:
        assert not window.frozen()
        close()
        assert window.frozen(), "close() left the window open"
    window.reopen()
    with window.tuning_window():
        window.seal()
    assert window.frozen() and window.sealed(), "a seal inside a window was undone"


class _EveryStep(RetuneKernels):
    """The broken variant: retunes after every step and never seals."""

    def on_step_end(self, state, out) -> None:
        self.retune()
        self.agree()


def check_retune_steps(cls) -> None:
    """Steps 1..2 retune (queued problems pending), step 2 seals; steps 3..5 do nothing."""
    order: list[str] = []
    cb = cls(order.append, anywhere=lambda n: n > 0, steps=2)
    cb.pending = lambda: 1
    cb.retune = lambda: order.append("retune")
    cb.agree = lambda: order.append("agree")
    for _ in range(5):
        cb.on_step_end(None, None)
    assert order == ["retune", "retuned", "agree"] * 2, order
    assert window.sealed() and window.frozen(), "the retune steps ended unsealed"


def test_retune_over_the_first_steps_only():
    check_retune_steps(RetuneKernels)


def test_negative_a_retune_every_step():
    must_fail(lambda: check_retune_steps(_EveryStep), "a retune every step")


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    tuner._TABLES.clear()
    tuner.reset()
    yield
    tuner._TABLES.clear()
    tuner.reset()


def check_sealed_queue() -> None:
    """A sealed op binds the kernel for a new problem and queues nothing."""
    op = choice.kernel_or_compile("test", "situ_glu", ffn.situ_glu, choice.shape_key)
    op.impls["compile"] = ops.CompiledOp("situ_glu", True, "eager")
    tuner.set_mode("table")
    window.seal()
    op(torch.randn(64, 32))
    assert not op.pending, f"{len(op.pending)} problem(s) queued after the seal"
    assert choice.pending_count() == 0


@pytest.mark.budget(60)
def test_sealed_ops_queue_nothing(fresh):
    check_sealed_queue()


@pytest.mark.budget(60)
def test_negative_queued_after_the_seal(fresh, monkeypatch):
    monkeypatch.setattr(choice, "sealed", lambda: False)
    must_fail(check_sealed_queue, "queued after the seal")


class _Generate:
    """A generate that records whether the tuner is frozen before each sampler step."""

    def __init__(self):
        self.seen: list[bool] = []

    def generate(self, steps=3, step_callback=None, **settings):
        for i in range(steps):
            self.seen.append(window.frozen())
            if step_callback is not None:
                step_callback(i, None)
        return SimpleNamespace(images=[])


class _EveryGrid(PipelineImages):
    """The broken variant: every sample grid opens a window."""

    def __call__(self, **settings) -> list:
        return self._first(**settings)


def check_first_step_open(cls) -> None:
    """Frozen run: the first grid's first sampler step is open, every later step and
    grid frozen; the user's step callback still runs."""
    window.freeze()
    gen, calls = _Generate(), []
    images = cls(gen)
    images(steps=3, step_callback=lambda i, x: calls.append(i))
    images(steps=3)
    assert gen.seen == [False, True, True, True, True, True], gen.seen
    assert calls == [0, 1, 2], calls
    assert window.frozen()


def test_first_sampler_step_only():
    check_first_step_open(PipelineImages)


def test_negative_every_grid_open():
    must_fail(lambda: check_first_step_open(_EveryGrid), "every grid open")
