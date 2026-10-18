"""VRAM accounting: a nested peak window never loses the enclosing window's peak; the
bar's peak is the step's true peak across timeline phase marks (peak allocated / peak
reserved / the device's use from the driver, as nvidia-smi shows it); the breakdown
parts (other processes, context, non-torch, fragmented, cached, allocated) sum to the
device's use, with or without NVML."""

import logging

import pytest

from kohakumoss.memory.device_use import PARTS, DeviceUse
from kohakumoss.memory.timeline import VramTimeline
from kohakumoss.training.callbacks.monitor import Vram
from kohakumoss.training.callbacks.progress import ProgressBar
from kohakumoss.training.state import TrainState
from kohakumoss_kernels.hardware import memory
from tests._util.logs import collecting
from tests._util.negative import must_fail
from tests.training.run.test_progress import FakeBar

pytestmark = pytest.mark.cpu
GIB = 2**30


def gib(x: float) -> int:
    return int(x * GIB)


class FakeMem:
    """A torch.cuda-like allocator: live / peak allocated and reserved, device info."""

    def __init__(self, other=3.0, context=0.5, extra=1.5, split=0.0):
        self.alloc = self.peak = self.res = self.res_peak = 0
        self.total, self.other = gib(24), gib(other)
        self.context, self.extra, self.split = gib(context), gib(extra), gib(split)

    def set(self, alloc: float, reserved: float | None = None) -> None:
        self.alloc = gib(alloc)
        self.peak = max(self.peak, self.alloc)
        if reserved is not None:
            self.res = gib(reserved)
            self.res_peak = max(self.res_peak, self.res)

    def max_memory_allocated(self, d):
        return self.peak

    def memory_allocated(self, d):
        return self.alloc

    def max_memory_reserved(self, d):
        return self.res_peak

    def memory_reserved(self, d):
        return self.res

    def reset_peak_memory_stats(self, d):
        self.peak, self.res_peak = self.alloc, self.res

    def mem_get_info(self, d):
        used = self.other + self.context + self.extra + self.res
        return self.total - used, self.total

    def memory_stats(self, d):
        return {
            "inactive_split_bytes.all.current": self.split,
            "allocated_bytes.all.current": self.alloc,
            "reserved_bytes.all.current": self.res,
            "allocated_bytes.all.peak": self.peak,
            "reserved_bytes.all.peak": self.res_peak,
        }


def _equal(a, b) -> None:
    assert a == b, (a, b)


@pytest.fixture
def make(monkeypatch):
    """A fresh fake allocator behind the peak accumulator: (fake, timeline, device use)."""

    def build(nvml=True):
        fake = FakeMem()
        monkeypatch.setattr(memory, "MODULES", {"cuda": fake})
        monkeypatch.setattr(memory, "_OPEN", {})
        extra, fake.extra = fake.extra, 0
        nvml_used = (lambda d: fake.other) if nvml else (lambda d: None)
        use = DeviceUse("cuda", mem=fake, nvml_used=nvml_used)
        fake.extra = extra
        return fake, VramTimeline("cuda"), use

    return build


def test_nested_window_keeps_the_outer_peak(make):
    fake, _, _ = make()
    outer = memory.open_window("cuda")
    fake.set(9.0)
    fake.set(1.0)
    with memory.window("cuda") as inner:
        fake.set(2.0)
    assert inner.alloc == gib(2.0)
    assert memory.close_window(outer)[0] == gib(9.0)
    raw = memory.open_window("cuda")
    fake.set(9.0)
    fake.set(1.0)
    fake.reset_peak_memory_stats(None)
    fake.set(2.0)
    cut = memory.close_window(raw)[0]
    must_fail(lambda: _equal(cut, gib(9.0)), "a raw reset inside the window")


def test_bar_peak_survives_phase_resets(make):
    fake, timeline, use = make()
    FakeBar.made = []
    pb = ProgressBar(3, "cuda", timeline, use, make=FakeBar, clock=lambda: 0.0)
    fake.set(9.6, 10.7)
    fake.set(2.0)
    timeline.mark("te")
    fake.set(3.9)
    today = f"{fake.max_memory_allocated(None) / GIB:.1f}"
    pb.on_step_end(TrainState(step=1, loss=1.0), None)
    assert FakeBar.made[-1].postfix["vram"] == "9.6/10.7/15.7G"
    must_fail(lambda: _equal(today, "9.6"), "the peak since the last phase reset")
    fake.set(5.0)
    pb.on_step_end(TrainState(step=2, loss=1.0), None)
    assert FakeBar.made[-1].postfix["vram"] == "5.0/10.7/15.7G"


def _sums(parts: dict[str, int], names=PARTS) -> None:
    assert sum(parts[p] for p in names) == parts["device"], parts


def _driver(fake) -> int:
    free, total = fake.mem_get_info(None)
    return total - free


def test_breakdown_sums_to_device_use(make):
    fake, _, use = make()
    fake.set(8.0, 10.7)
    fake.split = gib(0.6)
    parts = use.breakdown()
    _sums(parts)
    assert parts["device"] == gib(3.0) + gib(0.5) + gib(1.5) + gib(10.7)
    assert parts["other"] == gib(3.0) and parts["context"] == gib(0.5)
    assert parts["non_torch"] == gib(1.5)
    assert parts["fragmented"] == gib(0.6) and parts["allocated"] == gib(8.0)
    without = [p for p in PARTS if p != "non_torch"]
    must_fail(lambda: _sums(parts, without), "reserved taken for the device use")
    fake2, _, blind = make(nvml=False)
    fake2.set(1.0, 2.0)
    folded = blind.breakdown()
    _sums(folded)
    assert folded["other"] == 0 and folded["context"] == gib(3.5)
    assert folded["non_torch"] == gib(1.5)
    _equal(blind.used(), _driver(fake2))
    background = blind.used() - gib(3.5)
    must_fail(lambda: _equal(background, _driver(fake2)), "the context left out")


def test_breakdown_logs_and_metrics(make):
    fake, timeline, use = make()
    fake.set(4.0, 6.0)
    with collecting(logging.INFO) as logs:
        use.log(logging.getLogger("kohakumoss.vram.test"), "after build")
    table = logs.messages[-1]
    head = (
        "vram after build on cuda: 11.00 GiB on the device (other processes measured)"
    )
    assert table.startswith(head), table
    for part in PARTS:
        assert f"\n  {part}" in table, (part, table)
    state = TrainState()
    Vram(1, timeline, use).on_step_end(state, None)
    assert state.metrics["vram_device"] == pytest.approx(11.0)
    assert state.metrics["vram_other"] == pytest.approx(3.0)
    assert state.metrics["vram_non_torch"] == pytest.approx(1.5)
    assert state.metrics["vram_peak_reserved"] == pytest.approx(6.0)
