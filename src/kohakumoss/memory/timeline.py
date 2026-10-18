"""Device-memory timeline: per-phase peak and post-phase allocation (CUDA / XPU).

`mark(phase)` closes the phase that just ran: the peak allocated bytes of its peak
window and the bytes still allocated now. Three windows are open at a time (the phase,
the step for `take_step()`, the report for `report()`), all on the process-wide peak
accumulator (`kohakumoss_kernels.hardware.memory`), so no other measurement (a generate,
a prewarm replay, an encoder pass) cuts them. Every read is host-side allocator state.
"""

import torch

from kohakumoss_common.units import gib
from kohakumoss_kernels.hardware.memory import open_window, restart_window


class VramTimeline:
    """Phases of a step, e.g. `te` (text-encoder pass, buffers released at its end) and
    `step` (DiT forward / backward / optimiser)."""

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self.peaks: dict[str, int] = {}
        self.after: dict[str, int] = {}
        self.phase = open_window(self.device)
        self.step = open_window(self.device)
        self.window = open_window(self.device)

    def mark(self, phase: str) -> None:
        alloc, _ = restart_window(self.phase)
        self.peaks[phase] = max(self.peaks.get(phase, 0), alloc)
        self.after[phase] = self.phase.alloc

    def take_step(self) -> tuple[int, int]:
        """(peak allocated, peak reserved) bytes since the last call."""
        return restart_window(self.step)

    def report(self) -> dict[str, float]:
        """GiB per phase (`vram_<phase>_peak`, `vram_<phase>_after`) and overall since
        the last report (`vram_peak_alloc`, `vram_peak_reserved`)."""
        out = {f"vram_{k}_peak": gib(v) for k, v in self.peaks.items()}
        out.update({f"vram_{k}_after": gib(v) for k, v in self.after.items()})
        alloc, held = restart_window(self.window)
        out["vram_peak_alloc"] = gib(alloc)
        out["vram_peak_reserved"] = gib(held)
        self.peaks.clear()
        self.after.clear()
        return out


class NoTimeline:
    """Placeholder when the device has no memory counters (CPU)."""

    def mark(self, phase: str) -> None:
        return None

    def take_step(self) -> tuple[int, int]:
        return 0, 0

    def report(self) -> dict[str, float]:
        return {}


def build_timeline(device: torch.device | str):
    device = torch.device(device)
    return VramTimeline(device) if device.type in ("cuda", "xpu") else NoTimeline()
