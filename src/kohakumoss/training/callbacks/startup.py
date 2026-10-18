"""Step-1 time breakdown: where the first step's wall time went. Counter deltas (text /
VAE encode seconds, kernel tuning seconds and problems, torch.compile seconds and
graphs, allocator segments) and a sampled attribution of every thread's stack
(`sampler.CATEGORIES`: compile, kernel tuning, compile / fp8 candidate probes, Triton /
TileLang JIT loads, weight-stream setup and waits, encodes, data, optimizer, forward /
backward), the rest by innermost frame. Logged as tables and
written to `<output>/startup/step1_breakdown.json` and the step's metrics."""

import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from kohakumoss.logging.log import get_logger, log_table
from kohakumoss.training.callbacks.sampler import StackSampler, main_thread_name
from kohakumoss_kernels.tuning import tuner

log = get_logger(__name__)
TOP = 8


OUTER_COMPILE = ("entire_frame_compile", "_compile.compile_inner", "compile_inner")


def _compile_secs() -> float:
    """Seconds torch.compile spent so far in this process: its outermost per-frame
    timer (the nested phase timers overlap it), else the sum of all timers."""
    utils = getattr(torch._dynamo, "utils", None)
    metrics = getattr(utils, "compilation_time_metrics", {}) if utils else {}
    for key in OUTER_COMPILE:
        if key in metrics:
            return float(sum(metrics[key]))
    return float(sum(sum(v) for v in metrics.values()))


def _graphs() -> int:
    utils = getattr(torch._dynamo, "utils", None)
    counters = getattr(utils, "counters", None) if utils else None
    return int(counters["stats"]["unique_graphs"]) if counters is not None else 0


def _segments(device: torch.device) -> int:
    """Allocator segments (driver allocations) made on `device` so far."""
    if device.type not in ("cuda", "xpu"):
        return 0
    stats = torch.get_device_module(device).memory_stats(device)
    return int(stats.get("segment.all.allocated", 0))


def _secs(stats_owner) -> float:
    return float(getattr(stats_owner, "stats", {}).get("seconds", 0.0))


def attribution(samples: dict[tuple[str, str], float], main: str) -> dict:
    """{"main": {category: s}, "other": {frame: s} (top), "threads": {name: {category:
    s}}} of a sampler report; `other:*` categories of the main thread are folded into
    `other` and listed by frame."""
    main_cats: dict[str, float] = defaultdict(float)
    frames: dict[str, float] = defaultdict(float)
    threads: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (thread, cat), secs in samples.items():
        if thread != main:
            threads[thread][cat.split(":")[0]] += secs
            continue
        if cat.startswith("other:"):
            main_cats["other"] += secs
            frames[cat[len("other:") :]] += secs
        else:
            main_cats[cat] += secs
    top = dict(sorted(frames.items(), key=lambda kv: -kv[1])[:TOP])
    return {
        "main": dict(sorted(main_cats.items(), key=lambda kv: -kv[1])),
        "other": top,
        "threads": {t: dict(c) for t, c in threads.items()},
    }


class StepOneBreakdown:
    """Snapshots the counters and starts the stack sampler when built (just before
    step 1); at the end of step 1 (after the step's retune) reports both."""

    every = 1

    def __init__(
        self, text, vae, output_dir: str, device, clock=time.perf_counter, sample=True
    ):
        self.text, self.vae = text, vae
        self.root = Path(output_dir) / "startup"
        self.device = torch.device(device)
        self.clock = clock
        self.done = False
        self.start = self._snapshot()
        self.sampler = StackSampler().start() if sample else None

    def _snapshot(self) -> dict:
        return {
            "wall": self.clock(),
            "text_encode": _secs(self.text),
            "vae_encode": _secs(self.vae),
            "kernel_tuning": sum(tuner.TIMED_SECS.values()),
            "tuned_problems": sum(tuner.TIMED.values()),
            "torch_compile": _compile_secs(),
            "compiled_graphs": _graphs(),
            "alloc_segments": _segments(self.device),
        }

    def on_step_end(self, state, out) -> None:
        if self.done:
            return
        self.done = True
        end = self._snapshot()
        delta = {k: end[k] - self.start[k] for k in end}
        timed = ("text_encode", "vae_encode", "kernel_tuning", "torch_compile")
        delta["other"] = delta["wall"] - sum(delta[k] for k in timed)
        rows = [(k, round(v, 3)) for k, v in delta.items()]
        log_table(log, "step 1 counters (seconds; counts)", ("part", "value"), rows)
        record = {"counters": delta}
        if self.sampler is not None:
            self.sampler.stop()
            sampled = attribution(self.sampler.report(), main_thread_name())
            record["sampled"] = sampled
            main = [(k, round(v, 2)) for k, v in sampled["main"].items()]
            log_table(log, "step 1 main thread (sampled s)", ("category", "s"), main)
            top = [(k, round(v, 2)) for k, v in sampled["other"].items()]
            log_table(log, "step 1 'other' by frame (sampled s)", ("frame", "s"), top)
            for k, v in sampled["main"].items():
                state.metrics[f"step1_sampled_{k}"] = float(v)
        state.metrics.update({f"step1_{k}": float(v) for k, v in delta.items()})
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "step1_breakdown.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
