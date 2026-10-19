"""Between-step allocator release and accounting (CUDA / XPU): STEP_RELEASE "none"
(default) keeps the allocator's cached blocks; "empty_cache" returns the free cached
blocks after every optimizer step and an integer N after every N steps; the step
metrics report what the allocator returned. See
docs/internals/offload.md#between-step-release."""

import time
from functools import partial

import torch

from kohakumoss_common.units import GIB
from kohakumoss_kernels.hardware.memory import reserved, trim

DEFAULT = "none"
RELEASE_NAMES = {"none": 0, "empty_cache": 1}


def release_every(value) -> int:
    """STEP_RELEASE as a step period: 0 for "none", 1 for "empty_cache", N for
    an integer N >= 1."""
    if isinstance(value, str):
        return RELEASE_NAMES[value]
    return int(value)


def release_problems(value) -> list[str]:
    """What is wrong with a STEP_RELEASE value (empty = valid)."""
    if isinstance(value, str) and value in RELEASE_NAMES:
        return []
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return []
    return [
        f"STEP_RELEASE={value!r} not in {tuple(RELEASE_NAMES)} or an int >= 1 "
        "(every N steps)"
    ]


class StepRelease:
    """Every `every` steps, after the optimizer step: the device's free cached blocks
    go back to the driver (`empty(device)`, default `trim`: a synchronise +
    empty_cache); keeps what its last release returned (`released` bytes, host
    seconds)."""

    def __init__(self, device, every: int, empty=trim, clock=time.perf_counter):
        self.device = torch.device(device)
        self.every = every
        self.empty = empty
        self.clock = clock
        self.released, self.secs = 0, 0.0

    def on_step_end(self, state, out) -> None:
        began, before = self.clock(), reserved(self.device)
        self.empty(self.device)
        self.released = max(before - reserved(self.device), 0)
        self.secs = self.clock() - began

    def take(self) -> tuple[int, float]:
        """(bytes, host s) of the release since the last take, then zero."""
        out = self.released, self.secs
        self.released, self.secs = 0, 0.0
        return out


def _take_from(source: StepRelease, stats) -> tuple[int, float]:
    return source.take()


class CacheReleaseStats:
    """Every step: the bytes returned to the driver at that step, reserved before minus
    reserved after: the `source` release's (a StepRelease), else the drop of the
    allocator's reserved bytes since the previous step end; the callback frees nothing.
    Metrics: `cache_released_gib` (GiB returned at this step), and averaged over the
    steps since the last metric row (`rows` cadence) `cache_releases` (share of steps
    in which memory went back) and `cache_release_time` (host s / step of the
    release)."""

    every = 1

    def __init__(self, device, rows: int, source: StepRelease | None = None):
        self.device = torch.device(device)
        self.rows = max(int(rows), 1)
        self.mark = reserved(self.device)
        self._take = (
            CacheReleaseStats._drop if source is None else partial(_take_from, source)
        )
        self.first = True
        self._clear()

    def _clear(self) -> None:
        self.secs, self.count, self.steps = 0.0, 0, 0

    def _drop(self) -> tuple[int, float]:
        now = reserved(self.device)
        freed, self.mark = max(self.mark - now, 0), now
        return freed, 0.0

    def on_step_end(self, state, out) -> None:
        freed, secs = self._take(self)
        self.count += freed > 0
        self.secs += secs
        self.steps += 1
        state.metrics.update(
            cache_release_time=self.secs / self.steps,
            cache_releases=self.count / self.steps,
            cache_released_gib=freed / GIB,
        )
        if self.first or state.step % self.rows == 0:
            self.first = False
            self._clear()


def release_callbacks(device, rows: int, release=DEFAULT) -> list:
    """On an accelerator: the STEP_RELEASE callback (resolved once; none for "none")
    ahead of the accounting of what the allocator returned (metric rows every `rows`
    steps); nothing on the host."""
    device = torch.device(device)
    if device.type not in ("cuda", "xpu"):
        return []
    every = release_every(release)
    releasing = [StepRelease(device, every)] if every else []
    source = releasing[0] if releasing else None
    return releasing + [CacheReleaseStats(device, rows, source)]
