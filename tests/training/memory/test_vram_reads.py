"""The progress bar's per-step VRAM reads: the timeline's phase mark and step window
restart each read the allocator statistics once, and the driver's device use is
queried at most once a second (model-free: a counting allocator)."""

import pytest

from kohakumoss.memory.timeline import VramTimeline
from kohakumoss.training.callbacks.progress import ProgressBar
from kohakumoss.training.state import TrainState
from kohakumoss_kernels.hardware import memory
from tests._util.negative import must_fail
from tests.training.memory.test_vram_accounting import FakeMem
from tests.training.run.test_progress import FakeBar

pytestmark = pytest.mark.cpu


class CountingMem(FakeMem):
    """FakeMem counting every statistics read."""

    def __init__(self):
        super().__init__()
        self.reads = 0

    def memory_stats(self, d):
        self.reads += 1
        return super().memory_stats(d)


class Use:
    def __init__(self):
        self.calls = 0

    def used(self) -> int:
        self.calls += 1
        return 0


@pytest.fixture
def mem(monkeypatch):
    fake = CountingMem()
    monkeypatch.setattr(memory, "MODULES", {"cuda": fake})
    monkeypatch.setattr(memory, "_OPEN", {})
    return fake


def _old_mark(self, phase: str) -> None:
    after = memory.allocated(self.device)
    alloc, _ = memory.close_window(self.phase)
    self.phase = memory.open_window(self.device)
    self.peaks[phase] = max(self.peaks.get(phase, 0), alloc)
    self.after[phase] = after


def check_reads(mem: CountingMem) -> None:
    timeline = VramTimeline("cuda")
    mem.set(3.0, 4.0)
    before = mem.reads
    timeline.mark("step")
    timeline.take_step()
    assert mem.reads - before == 2, mem.reads - before
    assert timeline.after["step"] == mem.alloc


def test_one_statistics_read_per_window(mem, monkeypatch):
    check_reads(mem)
    monkeypatch.setattr(VramTimeline, "mark", _old_mark)
    must_fail(lambda: check_reads(mem), "close + open per mark")


def _bar_calls(use_every: float, mem) -> int:
    ticks = iter([0.0, 0.1, 0.2, 0.3, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2])
    use = Use()
    bar = ProgressBar(
        3,
        "cuda",
        VramTimeline("cuda"),
        use,
        make=FakeBar,
        clock=lambda: next(ticks),
        use_every=use_every,
    )
    for step in (1, 2, 3):
        bar.on_step_end(TrainState(step=step, loss=1.0), None)
    return use.calls


def test_device_use_on_the_refresh_cadence(mem):
    assert _bar_calls(1.0, mem) == 2
    must_fail(lambda: _equal(_bar_calls(0.0, mem), 2), "a driver query every step")


def _equal(a, b) -> None:
    assert a == b, (a, b)
