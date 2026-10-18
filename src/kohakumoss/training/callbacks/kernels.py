"""Kernel step callbacks: the between-step retune of uncovered problems, the step-1
prewarm-vs-training VRAM report, the step-1 vs steps 2-10 speed report and the switch
of the decision log to live mode. See docs/internals/tuning.md#prewarm."""

import statistics
import time
import weakref

import torch

from kohakumoss.logging.log import get_logger, hl
from kohakumoss.parallel.launch import dist_rank, ranks_meet
from kohakumoss.tuning.prewarm import PREWARM_PEAK, tunes
from kohakumoss_common.calls import nothing
from kohakumoss_common.units import GIB
from kohakumoss_kernels.backends.compile.choice import pending_count, retune_pending
from kohakumoss_kernels.hardware.memory import close_window, open_window
from kohakumoss_kernels.tuning import decisions
from kohakumoss_kernels.tuning.tuner import agree, measured_anywhere

log = get_logger(__name__)
STEP_ONE: "weakref.WeakSet" = weakref.WeakSet()


def late_prewarm(device: torch.device, secs: float) -> None:
    """A prewarm ran inside step 1 (`TextPrewarm`): the step-1 reports leave it out."""
    for cb in list(STEP_ONE):
        cb.prewarmed(device, secs)


class PrewarmVram:
    """At step 1: the prewarm's device peak against step 1's (log line and metrics
    `prewarm_peak_alloc` / `prewarm_peak_reserved`, GiB). Step 1's peak is a peak
    window opened when the callback is built (after the prewarm) and reopened after a
    prewarm inside step 1, so the TE / VAE phase marks inside the step cannot cut it."""

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
    """After every step, on every rank: when some rank has queued kernel-vs-compile
    problems (`anywhere(pending())`), time them (`retune_pending`) and meet on the store
    (`meet`); then bind the configs any rank measured on every rank (`tuner.agree`)."""

    every = 1

    def __init__(self, meet=ranks_meet, anywhere=None):
        self.retune = retune_pending
        self.pending = pending_count
        self.anywhere = anywhere or pending_anywhere
        self.meet = meet
        self.agree = agree

    def on_step_end(self, state, out) -> None:
        if self.anywhere(self.pending()):
            self.retune()
            self.meet("retuned")
        self.agree()


def pending_anywhere(count: int) -> bool:
    """Whether any rank has queued retune problems (`count` here; one all-reduce under
    a process group of two or more ranks)."""
    if dist_rank()[1] < 2:
        return count > 0
    return measured_anywhere(count)


class StartupSpeed:
    """Logs step 1's s/step against the median of steps 2..`last` (the prewarm's
    benefit shows as a step 1 close to the rest); a prewarm inside step 1 is left
    out of it."""

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
    (a later measured decision logs one line); a no-op on every later step."""

    every = 1

    def on_step_end(self, state, out) -> None:
        decisions.go_live()
        self.on_step_end = nothing


def retune_callbacks(k: dict) -> list:
    """The between-step retune, when TUNE times kernels."""
    return [RetuneKernels()] if tunes(k) else []


def vram_callbacks(k: dict, device: torch.device) -> list:
    """The step-1 prewarm-vs-training VRAM report, when TUNE times kernels on an
    accelerator."""
    accel = torch.device(device).type in ("cuda", "xpu")
    return [PrewarmVram(torch.device(device))] if tunes(k) and accel else []


def speed_callbacks(k: dict) -> list:
    """The step-1 vs steps 2-10 speed report (runs of two steps or more)."""
    last = min(10, k["STEPS"])
    return [StartupSpeed(last)] if last >= 2 else []
