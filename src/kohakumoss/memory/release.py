"""Between-step allocator release and accounting (CUDA / XPU): STEP_RELEASE "auto"
(default) releases on CUDA without expandable segments only once the cache holds more
than AUTO_MARGIN above the step's allocated peak, and never elsewhere; "none" returns no
memory to the driver between steps, "empty_cache" returns the free cached blocks after
every optimizer step, an integer N after every N steps; the step metrics report what
the allocator returned. See docs/internals/offload.md#between-step-release."""

import sys
import time

import torch

from kohakumoss_kernels.hardware.memory import (
    device_total,
    open_window,
    reserved,
    restart_window,
    trim,
)

GIB = 2**30
AUTO = "auto"
AUTO_SHARE = 0.10
AUTO_FLOOR = GIB
EXPANDABLE = "expandable_segments:True"
RELEASE_NAMES = {"none": 0, "empty_cache": 1}
_ALLOCATOR_SETTINGS = getattr(torch._C, "_accelerator_getAllocatorSettings", None)


def release_every(value) -> int:
    """STEP_RELEASE as a step period: 0 (never) for "none", 1 for "empty_cache", N for
    an integer N >= 1."""
    if isinstance(value, str):
        return RELEASE_NAMES[value]
    return int(value)


def release_problems(value) -> list[str]:
    """What is wrong with a STEP_RELEASE value (empty = valid)."""
    if isinstance(value, str) and (value in RELEASE_NAMES or value == AUTO):
        return []
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return []
    return [
        f"STEP_RELEASE={value!r} not in {(AUTO, *RELEASE_NAMES)} or an int >= 1 "
        "(every N steps)"
    ]


def expandable(device) -> bool:
    """The CUDA caching allocator on `device` runs expandable segments (the settings
    in force, read from torch; Linux only)."""
    if torch.device(device).type != "cuda" or _ALLOCATOR_SETTINGS is None:
        return False
    return sys.platform == "linux" and EXPANDABLE in _ALLOCATOR_SETTINGS()


def fragments(device) -> bool:
    """`device`'s allocator grows its reserved bytes step over step (CUDA without
    expandable segments)."""
    return torch.device(device).type == "cuda" and not expandable(device)


def auto_margin(device) -> int:
    """Bytes the cache may hold above the step's allocated peak under "auto": the larger
    of AUTO_SHARE of the device and AUTO_FLOOR."""
    return max(int(AUTO_SHARE * device_total(device)), AUTO_FLOOR)


class _Releaser:
    """A between-step release that keeps what its last `release(device)` returned:
    `released` bytes (reserved before minus reserved after) and its host seconds."""

    def __init__(self, device, release=trim, clock=time.perf_counter):
        self.device = torch.device(device)
        self.release = release
        self.clock = clock
        self.released, self.secs = 0, 0.0

    def _release(self) -> None:
        began, before = self.clock(), reserved(self.device)
        self.release(self.device)
        self.released = max(before - reserved(self.device), 0)
        self.secs = self.clock() - began

    def take(self) -> tuple[int, float]:
        """(bytes, host s) of the release since the last take, then zero."""
        out = self.released, self.secs
        self.released, self.secs = 0, 0.0
        return out


class AutoRelease(_Releaser):
    """Every step, after the optimizer step: when the allocator holds more than
    `margin` bytes above the step's allocated peak (one allocator read, no sync), the
    free cached blocks go back to the driver (`release(device)`)."""

    every = 1

    def __init__(self, device, margin: int, release=trim, clock=time.perf_counter):
        super().__init__(device, release, clock)
        self.margin = margin
        self.window = open_window(self.device)

    def on_step_end(self, state, out) -> None:
        peak, _ = restart_window(self.window)
        if reserved(self.device) - peak > self.margin:
            self._release()


class StepRelease(_Releaser):
    """Every `every` steps, after the optimizer step: the device's free cached blocks
    go back to the driver (`release(device)`, a synchronise + empty_cache)."""

    def __init__(self, device, every: int, release=trim, clock=time.perf_counter):
        super().__init__(device, release, clock)
        self.every = every

    def on_step_end(self, state, out) -> None:
        self._release()


class CacheReleaseStats:
    """Every step: the bytes returned to the driver at that step, reserved before minus
    reserved after: the `source` release's (a _Releaser), else the drop of the
    allocator's reserved bytes since the previous step end; the callback frees nothing.
    Metrics: `cache_released_gib` (GiB returned at this step), and averaged over the
    steps since the last metric row (`rows` cadence) `cache_releases` (share of steps
    in which memory went back) and `cache_release_time` (host s / step of the
    release)."""

    every = 1

    def __init__(self, device, rows: int, source: _Releaser | None = None):
        self.device = torch.device(device)
        self.rows = max(int(rows), 1)
        self.mark = reserved(self.device)
        self.take = self._drop if source is None else source.take
        self.first = True
        self._clear()

    def _clear(self) -> None:
        self.secs, self.count, self.steps = 0.0, 0, 0

    def _drop(self) -> tuple[int, float]:
        now = reserved(self.device)
        freed, self.mark = max(self.mark - now, 0), now
        return freed, 0.0

    def on_step_end(self, state, out) -> None:
        freed, secs = self.take()
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


def _releasing(device: torch.device, release) -> list:
    """The STEP_RELEASE callback for `release`, resolved once (empty: none)."""
    if release == AUTO:
        return [AutoRelease(device, auto_margin(device))] if fragments(device) else []
    every = release_every(release)
    return [StepRelease(device, every)] if every else []


def release_callbacks(device, rows: int, release=AUTO) -> list:
    """On an accelerator: the STEP_RELEASE callback (`release`, resolved once; none
    for "none", or for "auto" where the allocator does not fragment) ahead of the
    accounting of what it returned (metric rows every `rows` steps); nothing on the
    host."""
    device = torch.device(device)
    if device.type not in ("cuda", "xpu"):
        return []
    releasing = _releasing(device, release)
    source = releasing[0] if releasing else None
    return releasing + [CacheReleaseStats(device, rows, source)]
