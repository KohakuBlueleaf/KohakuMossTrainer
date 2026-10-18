"""Per-step sampler diagnostics: seconds of every step, and for a streamed DiT the H2D
bytes, copy-busy and copy-wait seconds of the step; logged at INFO as one line per step
and a summary when the batch ends. On an accelerator a step records one device event
and the batch's events are read once at its end (no sync between steps). See
docs/guides/inference.md#step-diagnostics."""

import time

import torch

from kohakumoss.logging.log import get_logger
from kohakumoss.offload.accel import module as accel
from kohakumoss.offload.accel import streamed
from kohakumoss.offload.weights.stream import read_timing

log = get_logger(__name__)


class _NoStream:
    def set_timing(self, on: bool) -> None:
        return None

    def cut_timing(self) -> dict:
        return {}


def _no_read(cut: dict) -> dict[str, float]:
    return {}


class _HostMarks:
    """Step marks on the host clock (CPU: the work is done when the call returns)."""

    def __init__(self, device: torch.device):
        self.device = device

    def mark(self) -> float:
        return time.perf_counter()

    def seconds(self, a: float, b: float) -> float:
        return b - a

    def settle(self) -> None:
        return None


class _DeviceMarks:
    """Step marks as timing events on the device's current stream."""

    def __init__(self, device: torch.device):
        self.device = device
        self.dm = accel(device)

    def mark(self):
        ev = self.dm.Event(enable_timing=True)
        ev.record(self.dm.current_stream(self.device))
        return ev

    def seconds(self, a, b) -> float:
        return a.elapsed_time(b) / 1e3

    def settle(self) -> None:
        self.dm.synchronize(self.device)


class StepClock:
    """`tick(i)` after sampler step `i` (one mark, no sync); `summary()` after a batch
    reads every step, logs them and the summary. `weight_stream` is the DiT's
    WeightStream or None; `rows` holds the per-step rows once `summary()` ran."""

    def __init__(self, device: torch.device, steps: int, weight_stream=None):
        self.device = device
        self.steps = steps
        self.weight_stream = weight_stream or _NoStream()
        self.read = _no_read if weight_stream is None else read_timing
        self.marks = _DeviceMarks(device) if streamed(device) else _HostMarks(device)
        self.rows: list[dict] = []
        self.ticks: list[tuple[int, object, dict]] = []

    def start(self) -> None:
        self.weight_stream.set_timing(True)
        self.weight_stream.cut_timing()
        self.t0 = self.marks.mark()
        self.ticks = []
        self.rows.clear()

    def tick(self, i: int) -> None:
        self.ticks.append((i, self.marks.mark(), self.weight_stream.cut_timing()))

    def _rows(self) -> list[dict]:
        self.marks.settle()
        rows, prev = [], self.t0
        for i, mark, cut in self.ticks:
            rows.append({"step": i + 1, "s": self.marks.seconds(prev, mark)})
            rows[-1].update(self.read(cut))
            prev = mark
        return rows

    def summary(self) -> dict:
        """The batch's per-step lines, then the mean seconds per step without the first
        and the copy / overlap ratios over the same steps."""
        self.weight_stream.set_timing(False)
        self.rows[:] = self._rows()
        for row in self.rows:
            log.info(_line(row, self.steps))
        rest = self.rows[1:] or self.rows
        mean = sum(r["s"] for r in rest) / max(len(rest), 1)
        out = {"first_s": self.rows[0]["s"] if self.rows else 0.0, "mean_s": mean}
        if rest and "copy_s" in rest[0]:
            copy = sum(r["copy_s"] for r in rest) / len(rest)
            wait = sum(r["wait_s"] for r in rest) / len(rest)
            out.update(
                h2d_gb=sum(r["h2d_gb"] for r in rest) / len(rest),
                copy_s=copy,
                wait_s=wait,
                overlap=1.0 - wait / copy if copy > 0 else 1.0,
            )
        log.info(_summary(out))
        return out


def _line(row: dict, steps: int) -> str:
    text = f"step {row['step']}/{steps}: {row['s']:.3f} s"
    if "copy_s" in row:
        text += (
            f", H2D {row['h2d_gb']:.2f} GB (copy {row['copy_s']:.3f} s busy, compute "
            f"waited {row['wait_s']:.3f} s), host syncs {row['host_syncs']}"
        )
    return text


def _summary(out: dict) -> str:
    text = f"steps: first {out['first_s']:.2f} s, then {out['mean_s']:.3f} s/step"
    if "copy_s" in out:
        text += (
            f"; streaming {out['h2d_gb']:.2f} GB/step, copy {out['copy_s']:.3f} s, "
            f"wait {out['wait_s']:.3f} s, overlap {out['overlap']:.0%}"
        )
    return text
