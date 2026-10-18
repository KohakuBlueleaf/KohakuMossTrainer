"""The tuner's working-memory bound, model-free (a fake allocator that honours the
per-process memory fraction): a config far beyond the leanest config of the same
problem is skipped before it allocates, the fastest config within the bound is stored
with every config's working memory, and the fraction is restored."""

from contextlib import contextmanager

import pytest
import torch

from kohakumoss_kernels.hardware import memory
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
DEV = torch.device("cuda", 0)
MIB = 1 << 20
GIB = 1 << 30
LEAN = {"mib": 100, "ms": 3.0}
FAT = {"mib": 15000, "ms": 1.0}
MID = {"mib": 600, "ms": 2.0}


class Allocator:
    """A torch.cuda-like allocator: a launch uses `nbytes` above the live bytes and
    raises out-of-memory past the per-process fraction."""

    def __init__(self):
        self.alloc = self.peak = self.res = 2 * GIB
        self.total = 24 * GIB
        self.fraction = None
        self.top = 0

    def memory_allocated(self, d):
        return self.alloc

    def max_memory_allocated(self, d):
        return self.peak

    def memory_reserved(self, d):
        return self.res

    def max_memory_reserved(self, d):
        return self.res

    def reset_peak_memory_stats(self, d):
        self.peak = self.alloc

    def mem_get_info(self, d):
        return self.total - self.res, self.total

    def set_per_process_memory_fraction(self, fraction, d):
        self.fraction = fraction

    def synchronize(self, d=None):
        return None

    def empty_cache(self):
        return None

    def use(self, nbytes: int) -> None:
        if (
            self.fraction is not None
            and self.alloc + nbytes > self.fraction * self.total
        ):
            raise torch.OutOfMemoryError("past the per-process memory fraction")
        self.peak = max(self.peak, self.alloc + nbytes)
        self.top = max(self.top, nbytes)


@pytest.fixture
def alloc(monkeypatch, tmp_path):
    fake = Allocator()
    monkeypatch.setattr(memory, "MODULES", {"cuda": fake})
    monkeypatch.setattr(memory, "_OPEN", {})
    tables = {}

    def table(kernel, device):
        return tables.setdefault(kernel, tuner.TuneTable(tmp_path / f"{kernel}.json"))

    monkeypatch.setattr(tuner, "table", table)
    monkeypatch.setattr(tuner, "_CHOSEN", {})
    monkeypatch.setattr(tuner, "TIMINGS", {"events": lambda fn, device: fn()})
    tuner.set_mode("table")
    fake.tables = tables
    return fake


def _launch(fake):
    def launch(config):
        def run():
            fake.use(config["mib"] * MIB)
            return config["ms"]

        return run

    return launch


def check_bound(fake, key: str) -> None:
    """The lean-first shortlist stores the fastest config within 8x the lean one's
    working memory; the fat config never allocates."""
    cfg = tuner.select(
        "test_working", DEV, key, lambda: [LEAN, FAT, MID], _launch(fake), shortlist=3
    )
    assert cfg == MID, cfg
    row = fake.tables["test_working"].rows[key]
    assert "working-memory bound" in row["failures"]["mib=15000,ms=1.0"], row
    assert row["working_mib"] == {"mib=100,ms=3.0": 100.0, "mib=600,ms=2.0": 600.0}
    assert fake.fraction == 1.0
    assert fake.top <= 8 * LEAN["mib"] * MIB, fake.top


def test_working_memory_bound(alloc):
    check_bound(alloc, "k")
    assert tuner.working_bound(None) is None
    assert tuner.working_bound(100 * MIB) == 800 * MIB
    assert tuner.working_bound(MIB) == tuner.WORKING_FLOOR


def test_negative_no_bound(alloc, monkeypatch):
    monkeypatch.setattr(tuner, "WORKING_FACTOR", 1000.0)
    must_fail(lambda: check_bound(alloc, "k2"), "every config unbounded")


BROKEN = {"mib": 50, "ms": 0.5, "raises": True}


def _raising(fake):
    def launch(config):
        def run():
            fake.use(config["mib"] * MIB)
            if config.get("raises"):
                raise RuntimeError("launch failed inside the bound")
            return config["ms"]

        return run

    return launch


def check_restored(fake, key: str) -> None:
    """A bounded config that raises last (a launch error; out of memory past the
    bound) leaves the process's memory fraction as it found it."""
    for last, launch in ((BROKEN, _raising(fake)), (FAT, _launch(fake))):
        fake.fraction = None
        cfg = tuner.select(
            "test_restore", DEV, f"{key}-{last['mib']}", lambda: [LEAN, last], launch
        )
        assert cfg == LEAN, cfg
        assert fake.fraction == 1.0, f"fraction left at {fake.fraction}"


@contextmanager
def _unrestored(device, bound=None):
    """The broken bound: the fraction is reset only when the block returns."""
    w = memory.Working()
    fake = memory.MODULES["cuda"]
    if bound is not None:
        w.enforced = True
        fake.set_per_process_memory_fraction((fake.res + bound) / fake.total, device)
    yield w
    fake.set_per_process_memory_fraction(1.0, device)


class Caching(Allocator):
    """An allocator holding `cached` free bytes until `empty_cache`."""

    def empty_cache(self):
        self.res = self.alloc


def _winner(monkeypatch, tmp_path, cached: int, name: str) -> dict:
    fake = Caching()
    fake.res = fake.alloc + cached
    monkeypatch.setattr(memory, "MODULES", {"cuda": fake})
    tab = tuner.TuneTable(tmp_path / f"{name}.json")
    monkeypatch.setattr(tuner, "table", lambda kernel, device: tab)
    monkeypatch.setattr(tuner, "_CHOSEN", {})
    tuner.select("test_cache", DEV, "k", lambda: [LEAN, FAT, MID], _launch(fake), 3)
    return tab.rows["k"]["config"]


def check_cache_free(monkeypatch, tmp_path, run: str) -> None:
    """The stored winner is the same whatever the allocator held cached (fresh tables
    per `run`, so nothing is read back)."""
    cold = _winner(monkeypatch, tmp_path, 0, f"cold-{run}")
    warm = _winner(monkeypatch, tmp_path, 20 * GIB, f"warm-{run}")
    assert cold == warm, (cold, warm)


def test_bound_independent_of_the_cache(alloc, monkeypatch, tmp_path):
    check_cache_free(monkeypatch, tmp_path, "trim")
    monkeypatch.setattr(tuner, "trim", lambda device: None)
    must_fail(
        lambda: check_cache_free(monkeypatch, tmp_path, "no-trim"), "the cache decides"
    )


def check_live_cap(fake, monkeypatch, cap) -> None:
    """After the prewarm (`set_cap`): a live key whose timing would pass the budget
    binds its first candidate untimed and unstored; the fat config never allocates;
    one trim follows the tune, none a later lookup."""
    trims = []
    monkeypatch.setattr(tuner, "trim", trims.append)
    monkeypatch.setattr(tuner, "DEFERRED", set())
    tuner.set_cap(DEV, cap)
    try:
        key = f"live-{cap}"
        got = tuner.select(
            "test_live", DEV, key, lambda: [LEAN, FAT, MID], _launch(fake), 3
        )
        assert got == LEAN, got
        assert fake.top < FAT["mib"] * MIB, fake.top / MIB
        assert (str(DEV), "test_live", key) in tuner.DEFERRED
        assert key not in fake.tables["test_live"].rows
        tuned = len(trims)
        assert tuned >= 1, "no trim after a live tune"
        tuner.select("test_live", DEV, key, lambda: [LEAN, FAT, MID], _launch(fake), 3)
        assert len(trims) == tuned, "a lookup trimmed"
    finally:
        tuner.set_cap(DEV, None)


def test_live_tunes_stay_within_the_budget(alloc, monkeypatch):
    check_live_cap(alloc, monkeypatch, 500 * MIB)
    must_fail(lambda: check_live_cap(alloc, monkeypatch, None), "no process cap")


def test_fraction_restored_when_a_config_raises(alloc, monkeypatch):
    check_restored(alloc, "r")
    monkeypatch.setattr(tuner, "working", _unrestored)
    must_fail(lambda: check_restored(alloc, "r2"), "no restore on an exception")
