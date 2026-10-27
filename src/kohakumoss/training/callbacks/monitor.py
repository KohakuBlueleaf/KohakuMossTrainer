"""Monitoring callbacks: throughput / MFU, VRAM, the console step line, metric rows,
instability dumps."""

import json
import math
import time
from pathlib import Path

from kohakumoss.logging.log import get_logger, hl
from kohakumoss.memory.device_use import NoDeviceUse
from kohakumoss.model.flops import forward_flops, peak_tflops, step_flops
from kohakumoss.training.state import TrainState
from kohakumoss_common.calls import nothing

log = get_logger(__name__)


class Throughput:
    """tokens/s, samples/s, step time and MFU between two log points."""

    edges = True

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
    """Peak allocated / reserved GiB since the last log point (CUDA / XPU devices),
    overall and per phase of the VRAM timeline (`te`: text-encoder passes, `step`: the
    rest; `_after` is what stays allocated when the phase ends), plus the device-use
    breakdown (`vram_device`, `vram_other`, `vram_context`, `vram_non_torch`,
    `vram_fragmented`, `vram_cached`, `vram_allocated`) when `device_use` is given."""

    edges = True

    def __init__(self, every: int, timeline, device_use=None):
        self.every = every
        self.timeline = timeline
        self.device_use = device_use if device_use is not None else NoDeviceUse()

    def on_step_end(self, state: TrainState, out) -> None:
        self.timeline.mark("step")
        state.metrics.update(self.timeline.report())
        state.metrics.update(self.device_use.metrics())


class VramAtStepOne:
    """Logs the device-use breakdown once, at the run's first step."""

    every = 1

    def __init__(self, device_use):
        self.device_use = device_use

    def on_step_end(self, state: TrainState, out) -> None:
        self.device_use.log(log, "after step 1")
        self.on_step_end = nothing


class TextEncodeStats:
    """Text-encoder share of the run between log points: seconds per step and of the
    wall time, tokens/s, host-to-device weight GB per step, TFLOP/s and MFU."""

    edges = True

    def __init__(self, every: int, encoder, device):
        self.every = every
        self.encoder = encoder
        self.peak = peak_tflops(device)
        self.t0 = time.perf_counter()
        self.mark = self._snapshot()
        self.step0 = 0

    def _snapshot(self) -> tuple:
        s = self.encoder.stats
        return (s["seconds"], s["tokens"], s.get("h2d_bytes", 0), s["passes"])

    def on_step_end(self, state: TrainState, out) -> None:
        now = time.perf_counter()
        dt = max(now - self.t0, 1e-9)
        cur = self._snapshot()
        sec, tok, h2d, passes = (a - b for a, b in zip(cur, self.mark))
        steps = max(state.step - self.step0, 1)
        flops = tok * self.encoder.flops_per_token
        state.metrics.update(
            te_time=sec / steps,
            te_share=sec / dt,
            te_tokens_per_s=tok / max(sec, 1e-9),
            te_h2d_gb=h2d / 1e9 / steps,
            te_passes=passes / steps,
            te_tflops=flops / max(sec, 1e-9) / 1e12,
        )
        if self.peak:
            state.metrics["te_mfu"] = flops / dt / 1e12 / self.peak
        self.t0, self.mark, self.step0 = now, cur, state.step


class VaeEncodeStats:
    """VAE share of the run between log points: seconds per step and of the wall time,
    images per step, host-to-device weight GB per step."""

    edges = True

    def __init__(self, every: int, vae):
        self.every = every
        self.vae = vae
        self.t0 = time.perf_counter()
        self.mark = self._snapshot()
        self.step0 = 0

    def _snapshot(self) -> tuple:
        s = self.vae.stats
        return (s["seconds"], s["images"], s.get("h2d_bytes", 0))

    def on_step_end(self, state: TrainState, out) -> None:
        now = time.perf_counter()
        cur = self._snapshot()
        sec, images, h2d = (a - b for a, b in zip(cur, self.mark))
        steps = max(state.step - self.step0, 1)
        state.metrics.update(
            vae_time=sec / steps,
            vae_share=sec / max(now - self.t0, 1e-9),
            vae_images=images / steps,
            vae_h2d_gb=h2d / 1e9 / steps,
        )
        self.t0, self.mark, self.step0 = now, cur, state.step


class FeedWait:
    """The encode feed between log points: seconds per step the rank waited for a
    ready batch (`feed_wait`; 0 while the feed keeps ahead) and the encoder-process
    seconds its batches took per step (`feed_encode`, text encoder + VAE)."""

    edges = True

    def __init__(self, every: int, feed):
        self.every = every
        self.feed = feed
        self.mark = self._snapshot()
        self.step0 = 0

    def _snapshot(self) -> tuple:
        busy = self.feed.text.stats["seconds"] + self.feed.vae.stats["seconds"]
        return self.feed.wait_seconds, busy

    def on_step_end(self, state: TrainState, out) -> None:
        cur = self._snapshot()
        wait, busy = (a - b for a, b in zip(cur, self.mark))
        steps = max(state.step - self.step0, 1)
        state.metrics.update(feed_wait=wait / steps, feed_encode=busy / steps)
        self.mark, self.step0 = cur, state.step


class StepLine:
    """The console step line (`step N [loss=, grad_norm=, lr=]`) every `every` steps and
    at the run's first and last step."""

    edges = True

    def __init__(self, every: int):
        self.every = every

    def on_step_end(self, state: TrainState, out) -> None:
        log.info(
            f"step {hl(state.step)}",
            loss=state.loss,
            grad_norm=state.grad_norm,
            lr=state.lr,
        )


class LogMetrics:
    """Pushes host metrics to the logger sinks every `every` steps (METRICS_EVERY) and at
    the run's first and last step (one host sync per row: the step's device metrics)."""

    edges = True

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
            "nonfinite_params": float(state.nonfinite_params),
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
        self,
        output_dir: str,
        factor: float = 5.0,
        beta: float = 0.98,
        warmup: int = 20,
        bundle=None,
    ):
        self.bundle = bundle
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
                "nonfinite_grads": list(getattr(self.bundle, "nonfinite_grads", [])),
            }
            (self.root / f"step_{state.step:08d}.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
        if math.isfinite(loss):
            self.avg = (
                loss
                if self.avg is None
                else self.beta * self.avg + (1 - self.beta) * loss
            )
            self.seen += 1
