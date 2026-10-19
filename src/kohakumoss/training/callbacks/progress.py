"""The training progress bar: one tqdm bar over the run's steps (rank 0), its postfix
updated every step from host values only. See docs/guides/logging.md."""

import math
import time

import torch

from kohakumoss.logging.log import bar
from kohakumoss.memory.device_use import NoDeviceUse
from kohakumoss.memory.timeline import NoTimeline
from kohakumoss.training.state import TrainState
from kohakumoss_common.units import GIB

POSTFIX = ("loss", "lr", "grad_norm", "s/it", "samples/s", "tokens/s", "vram")


def _no_vram(bar) -> str:
    return "-"


class ProgressBar:
    """tqdm over `steps` (ETA from tqdm): EMA-smoothed loss (`beta`), lr, grad norm,
    s/it, samples/s, tokens/s and `vram` = the step's peak allocated / peak reserved /
    the device's use (the driver's total - free, as nvidia-smi) in GiB (`timeline`
    closes the step's last phase and holds the per-step peaks across phase resets;
    `device_use` reads the device bytes from the driver at most every `use_every`
    seconds); closed after the last step. `enabled` False or a level above INFO shows
    nothing."""

    every = 1

    def __init__(
        self,
        steps: int,
        device,
        timeline=None,
        device_use=None,
        enabled: bool = True,
        beta: float = 0.9,
        clock=time.perf_counter,
        make=bar,
        use_every: float = 1.0,
    ):
        self.steps = steps
        self.device = torch.device(device)
        self.timeline = timeline if timeline is not None else NoTimeline()
        self.device_use = device_use if device_use is not None else NoDeviceUse()
        self._vram_of = _no_vram if timeline is None else ProgressBar._vram
        self.enabled = enabled
        self.beta = beta
        self.clock = clock
        self.make = make
        self.bar = None
        self.ema = None
        self.mark = None
        self.use_every = use_every
        self.use = (0, -math.inf)

    def _device_bytes(self) -> int:
        """The driver's used bytes, re-read once `use_every` seconds have passed."""
        value, when = self.use
        now = self.clock()
        if now - when >= self.use_every:
            self.use = value, when = self.device_use.used(), now
        return value

    def _vram(self) -> str:
        self.timeline.mark("step")
        alloc, held = self.timeline.take_step()
        use = self._device_bytes()
        return f"{alloc / GIB:.1f}/{held / GIB:.1f}/{use / GIB:.1f}G"

    def _open(self, state: TrainState) -> None:
        self.bar = self.make(
            total=self.steps,
            initial=state.step - 1,
            desc="train",
            unit="step",
            enabled=self.enabled,
        )
        self.mark = (self.clock(), state.samples, state.tokens)

    def on_step_end(self, state: TrainState, out) -> None:
        if self.bar is None:
            self._open(state)
        loss = state.loss
        if math.isfinite(loss):
            prev = loss if self.ema is None else self.ema
            self.ema = self.beta * prev + (1 - self.beta) * loss
        now = self.clock()
        t0, s0, n0 = self.mark
        dt = max(now - t0, 1e-9)
        post = {
            "loss": f"{self.ema:.4f}" if self.ema is not None else "nan",
            "lr": f"{state.lr:.2e}",
            "grad_norm": f"{state.grad_norm:.3f}",
            "s/it": f"{dt:.2f}",
            "samples/s": f"{(state.samples - s0) / dt:.1f}",
            "tokens/s": f"{(state.tokens - n0) / dt:.0f}",
            "vram": self._vram_of(self),
        }
        self.mark = (now, state.samples, state.tokens)
        self.bar.set_postfix(post, refresh=False)
        self.bar.update(1)
        if state.step >= self.steps:
            self.close()

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None
