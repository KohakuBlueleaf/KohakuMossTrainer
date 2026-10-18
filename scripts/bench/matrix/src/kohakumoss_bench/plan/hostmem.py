"""Host-RAM budget of a bench host: an estimate of what one training run pins on the
host (streamed base weights, offloaded activations, host optimizer state and gradients,
a process baseline) and a shared budget that holds a run back until its share fits, so
concurrent runs never oversubscribe the host. See docs/performance/bench-matrix.md#host-ram.
"""

import threading
import time
from pathlib import Path

from kohakumoss.memory.estimate import (
    GIB,
    PRECISION_BYTES,
    saved_per_token,
    total_params,
)
from kohakumoss.memory.profiles import PROFILES, checkpoint_mode
from kohakumoss.types.arch import PRESETS
from kohakumoss_bench.plan.config import Run

ARCH = PRESETS["base"]
BASELINE_GIB = 8.0
RAMP_S = 180.0
AUTO = "low"
MANUAL = {
    "CHECKPOINT": "checkpoint",
    "STREAM": "stream",
    "ACT_OFFLOAD": "act_offload",
    "OPTIM_STATE": "optim_state",
    "GRAD_OFFLOAD": "grad_offload",
}


def placement(knobs: dict) -> dict:
    """The memory-plan fields of a run's knobs: its profile (`auto` sized as `AUTO`),
    or the manual knobs, with MEMORY_OVERRIDES applied."""
    profile = knobs.get("MEMORY_PROFILE", "balanced")
    if profile == "manual":
        out = {f: knobs[k] for k, f in MANUAL.items() if knobs.get(k) is not None}
    else:
        out = dict(PROFILES.get(AUTO if profile == "auto" else profile, {}))
    out.update(knobs.get("MEMORY_OVERRIDES") or {})
    return out


def run_gib(run: Run) -> float:
    """Estimated host GiB a run holds beyond the shared weight stage."""
    k = run.knobs
    p = placement(k)
    tokens = int(k.get("MAX_TOKENS", 8192))
    full = k.get("ALGO") == "full"
    wbytes = PRECISION_BYTES.get(k.get("BASE_PRECISION") or "compute") or 2
    out = BASELINE_GIB * GIB
    if p.get("stream", "off") != "off":
        out += total_params(ARCH) * wbytes
    if p.get("act_offload"):
        mode, _ = checkpoint_mode(p.get("checkpoint", "block"))
        out += tokens * saved_per_token(ARCH, mode) * ARCH.depth * 2
    if full and p.get("optim_state") in ("stream", "cpu"):
        out += total_params(ARCH) * 16
    if full and p.get("grad_offload"):
        out += total_params(ARCH) * 4
    return out / GIB


def mem_available_gib() -> float | None:
    """MemAvailable of /proc/meminfo in GiB (what the host can still hand out, after
    every process's pinned, shared and resident use); None without /proc."""
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 2**20
    return None


class HostBudget:
    """Host RAM for a host's concurrent runs. `acquire` blocks until the run's share
    fits what is free now: MemAvailable minus `reserve_gib` minus the shares of runs
    granted in the last `RAMP_S` seconds (not yet resident); without /proc, `total_gib`
    minus the shares held. A run that fits nothing waits until no run of ours holds any.
    `release` returns the share."""

    def __init__(
        self, total_gib: float, stop, reserve_gib: float = 16.0, available=None
    ):
        self.total = total_gib
        self.reserve = reserve_gib
        self.available = available or mem_available_gib
        self.used = 0.0
        self.grants: list[tuple[float, float]] = []
        self.stop = stop
        self.cond = threading.Condition()

    def free(self) -> float:
        now = time.monotonic()
        self.grants = [(t, g) for t, g in self.grants if now - t < RAMP_S]
        avail = self.available()
        if avail is None:
            return self.total - self.used
        return avail - self.reserve - sum(g for _, g in self.grants)

    def acquire(self, gib: float) -> bool:
        """True once `gib` is held; False when the host stopped while waiting."""
        with self.cond:
            while self.used > 0 and gib > self.free():
                if self.stop.is_set():
                    return False
                self.cond.wait(10.0)
            self.used += gib
            self.grants.append((time.monotonic(), gib))
            return True

    def release(self, gib: float) -> None:
        with self.cond:
            self.used -= gib
            self.cond.notify_all()
