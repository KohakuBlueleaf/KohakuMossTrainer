"""Monitoring callbacks: throughput / MFU, VRAM, metric logging, instability dumps."""

import json
import math
import time
from pathlib import Path

import torch

from kohakumoss.training.flops import forward_flops, peak_tflops, step_flops
from kohakumoss.training.state import TrainState


class Throughput:
    """tokens/s, samples/s, step time and MFU between two log points."""

    def __init__(self, every: int, arch, mode: str, recompute: bool, device):
        self.every = every
        self.arch = arch
        self.mode = mode
        self.recompute = recompute
        self.peak = peak_tflops(device)
        self.t0 = time.perf_counter()
        self.mark = (0, 0, 0, 0, 0)

    def on_step_end(self, state: TrainState, out) -> None:
        now = time.perf_counter()
        dt = max(now - self.t0, 1e-9)
        tok, smp, sq, txt, step = (
            state.tokens - self.mark[0],
            state.samples - self.mark[1],
            state.joint_sq - self.mark[2],
            state.text_tokens - self.mark[3],
            state.step - self.mark[4],
        )
        fwd = (
            forward_flops(self.arch, tok, [], txt)
            + 4 * self.arch.dim * self.arch.depth * sq
        )
        flops = step_flops(fwd, self.mode, self.recompute)
        state.metrics.update(
            tokens_per_s=tok / dt,
            samples_per_s=smp / dt,
            step_time=dt / max(step, 1),
            tflops=flops / dt / 1e12,
        )
        if self.peak:
            state.metrics["mfu"] = flops / dt / 1e12 / self.peak
        self.t0 = now
        self.mark = (
            state.tokens,
            state.samples,
            state.joint_sq,
            state.text_tokens,
            state.step,
        )


class Vram:
    """Peak allocated / reserved GiB since the last log point (CUDA / XPU devices)."""

    def __init__(self, every: int, device):
        self.every = every
        self.device = device
        self.dm = torch.get_device_module(device)

    def on_step_end(self, state: TrainState, out) -> None:
        gib = 1024**3
        state.metrics.update(
            vram_peak_alloc=self.dm.max_memory_allocated(self.device) / gib,
            vram_peak_reserved=self.dm.max_memory_reserved(self.device) / gib,
        )
        self.dm.reset_peak_memory_stats(self.device)


class LogMetrics:
    """Pushes host metrics to the logger every `every` steps (one extra sync there)."""

    def __init__(self, every: int, logger):
        self.every = every
        self.logger = logger

    def on_step_end(self, state: TrainState, out) -> None:
        metrics = {
            "loss": state.loss,
            "grad_norm": state.grad_norm,
            "lr": state.lr,
            "loss_scale": state.scale,
            "skipped_steps": float(state.skipped),
            "nonfinite_steps": float(state.nonfinite),
        }
        for k, v in out.metrics.items():
            metrics[k] = float(v)
        metrics.update(state.metrics)
        self.logger.log(metrics, state.step)
        state.metrics.clear()


class InstabilityDump:
    """Writes `<output>/instability/step_X.json` on a skipped step or a loss spike."""

    every = 1

    def __init__(
        self, output_dir: str, factor: float = 5.0, beta: float = 0.98, warmup: int = 20
    ):
        self.root = Path(output_dir) / "instability"
        self.factor = factor
        self.beta = beta
        self.warmup = warmup
        self.avg = None
        self.seen = 0

    def on_step_end(self, state: TrainState, out) -> None:
        loss = state.loss
        spike = (
            self.avg is not None
            and self.seen >= self.warmup
            and math.isfinite(loss)
            and loss > self.factor * self.avg
        )
        if not state.ok or spike:
            self.root.mkdir(parents=True, exist_ok=True)
            record = {
                "step": state.step,
                "reason": "non-finite" if not state.ok else "spike",
                "loss": loss,
                "loss_avg": self.avg,
                "grad_norm": state.grad_norm,
                "loss_scale": state.scale,
                "lr": state.lr,
                "batch_ids": state.batch_ids,
                "metrics": {k: float(v) for k, v in out.metrics.items()},
            }
            (self.root / f"step_{state.step:08d}.json").write_text(
                json.dumps(record, indent=2)
            )
        if math.isfinite(loss):
            self.avg = (
                loss
                if self.avg is None
                else self.beta * self.avg + (1 - self.beta) * loss
            )
            self.seen += 1
