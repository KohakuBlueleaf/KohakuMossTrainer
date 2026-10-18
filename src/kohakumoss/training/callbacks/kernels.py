"""Kernel step callbacks: the between-step retune of uncovered problems over the first
steps (then the tuner is sealed), the step-1 prewarm-vs-training VRAM report, the
step-1 vs steps 2-10 speed report and the switch of the decision log to live mode
(with the tuner's deferral under a process group). See
docs/internals/tuning.md#prewarm."""

import statistics
import time
import weakref

import torch

from kohakumoss.logging.log import get_logger, hl
from kohakumoss.parallel.launch import dist_rank, ranks_meet
from kohakumoss.tuning.prewarm import PREWARM_PEAK, tunes
from kohakumoss_common.calls import nothing
from kohakumoss_common.units import GIB
from kohakumoss_kernels.backends.compile.choice import (
    drop_pending,
    pending_count,
    retune_pending,
)
from kohakumoss_kernels.hardware.memory import close_window, open_window
from kohakumoss_kernels.tuning import decisions
from kohakumoss_kernels.tuning.tuner import agree, positive_anywhere
from kohakumoss_kernels.tuning.window import seal

log = get_logger(__name__)
STEP_ONE: "weakref.WeakSet" = weakref.WeakSet()
RETUNE_STEPS = 3


def late_prewarm(device: torch.device, secs: float) -> None:
    """A prewarm ran inside step 1 (`TextPrewarm`): the step-1 reports leave it out."""
    for cb in list(STEP_ONE):
        cb.prewarmed(device, secs)


class PrewarmVram:
    """At step 1: the prewarm's device peak against step 1's (log line and metrics
    `prewarm_peak_alloc` / `prewarm_peak_reserved`, GiB). Step 1's peak is a peak
    window opened when the callback is built (after the prewarm) and reopened after a
    prewarm inside step 1."""

    every = 1

    def __init__(self, device: torch.device):
        self.device = device
        self.window = open_window(device)
        self.done = False
        STEP_ONE.add(self)

    def prewarmed(self, device: torch.device, secs: float) -> None:
        if not self.done and torch.device(device) == self.device:
            close_window(self.window)
            self.window = open_window(self.device)

    def on_step_end(self, state, out) -> None:
        if self.done:
            return
        self.done = True
        alloc, held = PREWARM_PEAK.get(str(self.device), (0, 0))
        step_alloc, step_held = close_window(self.window)
        state.metrics.update(
            {"prewarm_peak_alloc": alloc / GIB, "prewarm_peak_reserved": held / GIB}
        )
        log.info(
            f"VRAM peak: prewarm {hl(f'{alloc / GIB:.2f}')} GiB allocated / "
            f"{hl(f'{held / GIB:.2f}')} reserved; step 1 "
            f"{hl(f'{step_alloc / GIB:.2f}')} / {hl(f'{step_held / GIB:.2f}')}"
        )


class RetuneKernels:
    """After each of the process's first `steps` steps, on every rank: when some rank
    has queued kernel-vs-compile problems (`anywhere(pending())`), time them
    (`retune_pending`) and meet on the store (`meet`); then bind the configs any rank
    measured on every rank (`tuner.agree`). After the last of them (`close`): the
    queued problems are dropped and stay on the kernel, and `window.seal()` stops
    kernel-vs-compile queueing (the first sample grid's window can still time missing
    keys); a no-op on every later step."""

    every = 1

    def __init__(self, meet=ranks_meet, anywhere=None, steps: int = RETUNE_STEPS):
        self.retune = retune_pending
        self.pending = pending_count
        self.anywhere = anywhere or pending_anywhere
        self.meet = meet
        self.agree = agree
        self.steps = steps
        self.seen = 0

    def on_step_end(self, state, out) -> None:
        self.seen += 1
        if self.seen <= self.steps and self.anywhere(self.pending()):
            self.retune()
            self.meet("retuned")
        self.agree()
        if self.seen >= self.steps:
            self.close()

    def close(self) -> None:
        left = drop_pending()
        seal()
        self.on_step_end = nothing
        log.info(
            f"kernel-vs-compile retune closed after step {hl(self.seen)} of this process: "
            f"{hl(left)} queued kernel-vs-compile problem(s) left on the kernel"
        )


def pending_anywhere(count: int) -> bool:
    """Whether any rank has queued retune problems (`count` here; one all-reduce under
    a process group of two or more ranks)."""
    if dist_rank()[1] < 2:
        return count > 0
    return positive_anywhere(count)


class StartupSpeed:
    """Logs step 1's s/step against the median of steps 2..`last`; a prewarm inside
    step 1 is left out of it."""

    every = 1

    def __init__(self, last: int = 10, clock=time.perf_counter):
        self.last = last
        self.clock = clock
        self.marks = [clock()]
        STEP_ONE.add(self)

    def prewarmed(self, device: torch.device, secs: float) -> None:
        if len(self.marks) == 1:
            self.marks[0] += secs

    def on_step_end(self, state, out) -> None:
        if len(self.marks) > self.last:
            return
        self.marks.append(self.clock())
        if len(self.marks) - 1 == self.last:
            gaps = [b - a for a, b in zip(self.marks, self.marks[1:])]
            rest = statistics.median(gaps[1:])
            first, median = f"{gaps[0]:.3f} s/step", f"{rest:.3f} s/step"
            ratio = f"{gaps[0] / max(rest, 1e-9):.1f}x"
            log.info(
                f"step 1 {hl(first)}, steps 2-{self.last} median {hl(median)} "
                f"({hl(ratio)})"
            )


class DecisionsLive:
    """After step 1 (and its retune): the startup kernel-decision table, then live mode
    (a later measured decision logs one line) and, under a process group, the tuner's
    deferral (`tuner.defer_if_grouped`); a no-op on every later step."""

    every = 1

    def on_step_end(self, state, out) -> None:
        decisions.go_live()
        self.on_step_end = nothing


def retune_callbacks(k: dict) -> list:
    """The between-step retune of the first RETUNE_STEPS steps, when TUNE times
    kernels."""
    steps = k.get("RETUNE_STEPS", RETUNE_STEPS)
    return [RetuneKernels(steps=steps)] if tunes(k) else []


def vram_callbacks(k: dict, device: torch.device) -> list:
    """The step-1 prewarm-vs-training VRAM report, when TUNE times kernels on an
    accelerator."""
    accel = torch.device(device).type in ("cuda", "xpu")
    return [PrewarmVram(torch.device(device))] if tunes(k) and accel else []


def speed_callbacks(k: dict) -> list:
    """The step-1 vs steps 2-10 speed report (runs of two steps or more)."""
    last = min(10, k["STEPS"])
    return [StartupSpeed(last)] if last >= 2 else []
