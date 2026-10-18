"""The local accelerator as `kohakumoss-config new` sees it: kind, VRAM, bf16, count."""

from dataclasses import dataclass

import torch

from kohakumoss_common.units import gib
from kohakumoss_kernels.hardware.arch import arch_of


@dataclass(frozen=True)
class Accelerator:
    kind: str  # "cuda" | "xpu" | "cpu"
    vram_gib: float  # device 0, exact (presets.holds applies the tolerance)
    bf16: bool
    count: int


def detect() -> Accelerator:
    """Device 0 of the first accelerator kind present, else the CPU."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        bf16 = arch_of("cuda:0").has_bf16
        return Accelerator(
            "cuda", gib(props.total_memory), bf16, torch.cuda.device_count()
        )
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        props = torch.xpu.get_device_properties(0)
        return Accelerator(
            "xpu", gib(props.total_memory), True, torch.xpu.device_count()
        )
    return Accelerator("cpu", 0.0, False, 0)
