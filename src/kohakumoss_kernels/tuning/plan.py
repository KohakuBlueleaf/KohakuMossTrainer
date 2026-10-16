"""Analytical ranking: estimated time of a config from its work and the device rates."""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from kohakumoss_kernels.hardware.device import Device


@dataclass(frozen=True)
class Work:
    """What one config does: programs, MMA flops, exp count, bytes from DRAM."""

    programs: int
    mma_flops: float
    exp_ops: float
    dram_bytes: float
    resident: int = 1


def estimate_ms(dev: Device, work: Work, mma_eff: float = 0.55) -> float:
    """Roofline time scaled by wave quantisation: `programs` run `resident` per SM."""
    mma = dev.mma_tflops if not math.isnan(dev.mma_tflops) else 50.0
    dram = dev.dram_gbs if not math.isnan(dev.dram_gbs) else 500.0
    compute = (work.mma_flops + 16.0 * work.exp_ops) / (mma * 1e12 * mma_eff)
    memory = work.dram_bytes / (dram * 1e9)
    slots = max(1, dev.sms * max(1, work.resident))
    waves = math.ceil(work.programs / slots)
    fill = work.programs / (waves * slots)
    return 1e3 * max(compute, memory) / max(fill, 1e-3)


def rank(
    dev: Device, configs: Iterable[dict], work: Callable[[dict], Work]
) -> list[dict]:
    """Configs sorted by estimated time (best first)."""
    scored = [(estimate_ms(dev, work(c)), i, c) for i, c in enumerate(configs)]
    return [c for _, _, c in sorted(scored)]
