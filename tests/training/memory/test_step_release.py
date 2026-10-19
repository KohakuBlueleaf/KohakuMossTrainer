"""Between-step release and accounting, model-free: the stats callback returns nothing to
the driver, and the step metrics carry the bytes returned at that step (reserved before
minus reserved after); none on the host."""

import pytest

from kohakumoss.memory import release as step_release
from kohakumoss.memory.release import (
    CacheReleaseStats,
    StepRelease,
    release_callbacks,
    release_problems,
)
from kohakumoss.training.state import TrainState
from kohakumoss_kernels.hardware import memory
from tests._util.negative import must_fail
from tests.training.memory.test_vram_accounting import FakeMem

pytestmark = pytest.mark.cpu


class Counted(FakeMem):
    """FakeMem with the allocator's freed-bytes counter and a counting empty_cache."""

    def __init__(self):
        super().__init__()
        self.freed = 0
        self.emptied = 0

    def memory_stats(self, d):
        return {**super().memory_stats(d), "reserved_bytes.all.freed": self.freed}

    def empty_cache(self):
        self.emptied += 1

    def synchronize(self, d=None):
        return None


@pytest.fixture
def fake(monkeypatch):
    mem = Counted()
    monkeypatch.setattr(memory, "MODULES", {"cuda": mem})
    monkeypatch.setattr(memory, "_OPEN", {})
    return mem


class Releasing(CacheReleaseStats):
    """The broken variant: returns the cache to the driver after every step."""

    def on_step_end(self, state, out) -> None:
        memory.release(self.device)
        super().on_step_end(state, out)


class Cumulative(CacheReleaseStats):
    """The broken variant: reports the bytes returned since the start, not this step's."""

    total = 0

    def on_step_end(self, state, out) -> None:
        super().on_step_end(state, out)
        self.total += state.metrics["cache_released_gib"]
        state.metrics["cache_released_gib"] = self.total


def _scenario(fake, callback=CacheReleaseStats) -> None:
    """No release bound, four steps (metric rows every 2); reserved at each step end
    is 8, 6, 6, 5 GiB after 8 at the start: the steps return 0, 2, 0, 1 GiB."""
    fake.set(6.0, 8.0)
    cb = callback("cuda", 2)
    state = TrainState()
    for step, held in enumerate((8.0, 6.0, 6.0, 5.0), start=1):
        state.step = step
        fake.set(4.0, held)
        cb.on_step_end(state, None)
        assert state.metrics["cache_released_gib"] == pytest.approx(
            (0.0, 2.0, 0.0, 1.0)[step - 1]
        ), step
        if step == 2:
            assert state.metrics["cache_releases"] == 1.0
    assert state.metrics["cache_releases"] == 0.5
    assert fake.emptied == 0, f"{fake.emptied} releases between steps"


def test_no_release_between_steps(fake):
    _scenario(fake)


def test_negative_a_callback_that_releases(fake):
    must_fail(lambda: _scenario(fake, Releasing), "the cache returned every step")


def _released_steps(fake, callback=CacheReleaseStats) -> list[float]:
    """Two steps with a bound release: reserved grows to 10 then 9 GiB inside the step
    and the release brings it to 7 then 8 GiB; the per-step GiB reported."""
    after = iter((7.0, 8.0))
    ticks = iter(range(100))
    release = StepRelease(
        "cuda",
        1,
        empty=lambda d: fake.set(4.0, next(after)),
        clock=lambda: 1.0 * next(ticks),
    )
    cb = callback("cuda", 10, release)
    state, seen = TrainState(), []
    for step, held in enumerate((10.0, 9.0), start=1):
        state.step = step
        fake.set(6.0, held)
        release.on_step_end(state, None)
        cb.on_step_end(state, None)
        seen.append(state.metrics["cache_released_gib"])
        assert state.metrics["cache_release_time"] == 1.0
    return seen


def check_released(build) -> None:
    assert build() == pytest.approx([3.0, 1.0])


def test_released_bytes_are_this_steps_reserved_drop(fake):
    check_released(lambda: _released_steps(fake))


def test_negative_a_cumulative_release_counter(fake):
    must_fail(
        lambda: check_released(lambda: _released_steps(fake, Cumulative)),
        "the released bytes summed over the run",
    )


def test_a_step_without_its_release_reports_zero(fake):
    after = iter((7.0,))
    release = StepRelease("cuda", 2, empty=lambda d: fake.set(4.0, next(after)))
    cb = CacheReleaseStats("cuda", 1, release)
    state = TrainState()
    fake.set(6.0, 10.0)
    release.on_step_end(state, None)
    cb.on_step_end(state, None)
    assert state.metrics["cache_released_gib"] == pytest.approx(3.0)
    cb.on_step_end(state, None)
    assert state.metrics["cache_released_gib"] == 0.0


def test_release_stats_on_accelerators_only(fake):
    assert release_callbacks("cpu", 1) == []
    (cb,) = release_callbacks("cuda", 1, "none")
    assert isinstance(cb, CacheReleaseStats)
    release, stats = release_callbacks("cuda", 1, "empty_cache")
    assert stats._take.args == (release,), "the stats do not read the bound release"


def check_default(default) -> None:
    """The default binds no release inside the training loop, and "auto" is gone."""
    assert default == "none"
    assert _releases(release_callbacks("cuda", 1, default)) == []
    assert release_problems("auto"), "'auto' still accepted"


def test_default_releases_nothing(fake):
    check_default(step_release.DEFAULT)


def test_negative_a_releasing_default(fake):
    must_fail(lambda: check_default("empty_cache"), "a default that releases")


def _releases(callbacks: list) -> list:
    return [cb for cb in callbacks if isinstance(cb, StepRelease)]


def check_step_release(build) -> None:
    """STEP_RELEASE "none" binds no release; "empty_cache" one that empties once per
    step it runs; N one that runs every N steps; bad values are knob problems."""
    assert _releases(build("none")) == []
    (every,) = _releases(build("empty_cache"))
    assert every.every == 1
    calls = []
    every.empty = calls.append
    for _ in range(3):
        every.on_step_end(TrainState(), None)
    assert len(calls) == 3
    (third,) = _releases(build(3))
    assert third.every == 3
    assert release_problems("none") == release_problems(4) == []
    assert release_problems("sometimes") and release_problems(0)


def test_step_release_knob(fake):
    check_step_release(lambda value: release_callbacks("cuda", 1, value))


def test_negative_a_release_under_none(fake):
    def always(value):
        return [StepRelease("cuda", 1), *release_callbacks("cuda", 1, value)]

    must_fail(lambda: check_step_release(always), "a release bound under 'none'")
