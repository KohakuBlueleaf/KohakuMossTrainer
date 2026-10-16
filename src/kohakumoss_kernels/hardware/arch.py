"""Arch identification: maps a torch device to an `Arch` entry of the arch table."""

from dataclasses import dataclass

import torch

SM_TABLE = (70, 75, 80, 86, 89, 90, 100, 120)


@dataclass(frozen=True)
class Arch:
    """One arch-table entry. `name` is `smXX`, `xpu` or `cpu`."""

    name: str
    sm: int
    kind: str
    has_bf16: bool
    has_fp8: bool
    tensor_core_fp16: bool

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"


def _cuda_arch(major: int, minor: int) -> Arch:
    sm = major * 10 + minor
    entry = max((s for s in SM_TABLE if s <= sm), default=SM_TABLE[0])
    return Arch(
        name=f"sm{entry}",
        sm=entry,
        kind="cuda",
        has_bf16=entry >= 80,
        has_fp8=entry >= 89,
        tensor_core_fp16=entry >= 70,
    )


CPU = Arch("cpu", 0, "cpu", True, False, False)
XPU = Arch("xpu", 0, "xpu", True, False, True)


def arch_of(device: torch.device | str) -> Arch:
    """Arch-table entry of `device`; minor revisions inherit the closest lower entry."""
    device = torch.device(device)
    match device.type:
        case "cuda":
            major, minor = torch.cuda.get_device_capability(device)
            return _cuda_arch(major, minor)
        case "xpu":
            return XPU
        case _:
            return CPU
