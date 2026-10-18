"""Profiled windows of training steps on every rank: torch.profiler over steps
(first, last], the chrome trace on disk and its kernel timeline breakdown in the step
metrics; and the CUDA profiler API window an external profiler (ncu, nsys) records.
See docs/guides/multi-gpu.md (step breakdown)."""

import gzip
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from kohakumoss.logging.log import get_logger, hl
from kohakumoss.parallel.telemetry.timeline import breakdown
from kohakumoss_common.calls import nothing
from kohakumoss_kernels.hardware.current import synchronize

log = get_logger(__name__)


class StepTrace:
    """Profiles steps `first + 1 .. last` (the window opens at the end of step
    `first`); at the end of step `last` writes `<root>/trace.rank<r>.json.gz` and
    `<root>/trace.rank<r>.breakdown.json` and adds the `trace_*` breakdown (per step)
    of `device` to the step's metrics, and `trace_dev<i>_*` of every other device in
    `extra_devices` (the threaded pipeline's stages)."""

    every = 1

    def __init__(
        self, first: int, last: int, root: str, rank: int, device, extra_devices=()
    ):
        self.first, self.last = first, last
        self.root = Path(root)
        self.rank = rank
        self.device = torch.device(device)
        self.extra = [torch.device(d) for d in extra_devices]
        self.prof = None
        self.t0 = 0.0

    def on_step_end(self, state, out) -> None:
        if state.step == self.first:
            self._open()
        elif state.step == self.last and self.prof is not None:
            state.metrics.update(self._close())
            self.on_step_end = _done

    def _open(self) -> None:
        activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        self.prof = profile(activities=activities, record_shapes=False)
        for device in (self.device, *self.extra):
            synchronize(device)
        self.prof.start()
        self.t0 = time.perf_counter()

    def _close(self) -> dict:
        for device in (self.device, *self.extra):
            synchronize(device)
        wall_us = (time.perf_counter() - self.t0) * 1e6
        self.prof.stop()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"trace.rank{self.rank}.json.gz"
        self.prof.export_chrome_trace(str(path))
        getattr(self.prof, "wait_for_exports", nothing)()
        with gzip.open(path, "rt", encoding="utf-8") as f:
            events = json.load(f).get("traceEvents", [])
        steps = self.last - self.first
        out = breakdown(events, _index(self.device), wall_us, steps)
        for device in self.extra:
            i = _index(device)
            part = breakdown(events, i, wall_us, steps)
            out.update({f"trace_dev{i}_{k[6:]}": v for k, v in part.items()})
        out["trace_steps"] = float(steps)
        summary = self.root / f"trace.rank{self.rank}.breakdown.json"
        summary.write_text(json.dumps(out, indent=1), encoding="utf-8")
        log.info(f"trace of steps {self.first + 1}-{self.last}: {hl(path, 'path')}")
        self.prof = None
        return out


class ProfilerWindow:
    """`cudaProfilerStart` at the end of step `first`, `cudaProfilerStop` at the end of
    step `last`: an external profiler started with `--profile-from-start off` (ncu) or
    `--capture-range=cudaProfilerApi` (nsys) records steps `first + 1 .. last`. CUDA
    devices only."""

    every = 1

    def __init__(self, first: int, last: int, device):
        self.first, self.last = first, last
        self.device = torch.device(device)

    def on_step_end(self, state, out) -> None:
        if state.step == self.first:
            synchronize(self.device)
            torch.cuda.profiler.start()
        elif state.step == self.last:
            synchronize(self.device)
            torch.cuda.profiler.stop()
            self.on_step_end = _done


def _index(device: torch.device) -> int:
    return device.index if device.index is not None else 0


def _done(state, out) -> None:
    return None
