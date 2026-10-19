"""Device-generic stream / event access for the offload engines: CUDA and XPU run the
side-stream paths, every other device the synchronous host paths."""

import os
import sys

import torch

from kohakumoss_common.devices import streamed_type
from kohakumoss_kernels.hardware.current import with_index as with_index
from kohakumoss_kernels.hardware.memory import mem_info

EXPANDABLE = "expandable_segments:True"
ROUNDED = "roundup_power2_divisions:4"
ALLOC_ENV = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
HEADROOM = 0.92
_SET_ALLOCATOR = getattr(
    torch._C,
    "_accelerator_setAllocatorSettings",
    torch.cuda.memory._set_allocator_settings,
)


class DoneEvent:
    """An event that has already completed (host work: nothing to wait for)."""

    def synchronize(self) -> None:
        return None


DONE = DoneEvent()


def default_type() -> str:
    """The accelerator a run uses: cuda, else xpu, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def streamed(device: torch.device | str) -> bool:
    """The device has side streams, events and pinned host memory."""
    return streamed_type(torch.device(device).type)


def module(device: torch.device | str):
    """`torch.cuda` / `torch.xpu` for `device` (Stream, Event, stream, current_stream)."""
    return torch.get_device_module(torch.device(device))


def release_cache(device: torch.device | str) -> None:
    """Return the caching allocator's free blocks on `device` to the driver."""
    if streamed(device):
        module(device).empty_cache()


def release_host_cache() -> None:
    """Return the caching host allocator's free pinned blocks to the system."""
    if torch.accelerator.is_available():
        torch.accelerator.empty_host_cache()


def configure_allocator(device: torch.device | str) -> str:
    """Before the run's first allocation: expandable segments for the CUDA caching
    allocator where it has them (Linux), else power-of-2 rounding of request sizes
    (ROUNDED); a PYTORCH_CUDA_ALLOC_CONF / PYTORCH_ALLOC_CONF in the environment wins.
    Returns the settings in force. See docs/internals/offload.md#allocator."""
    device = torch.device(device)
    if device.type != "cuda":
        return f"{device.type} default"
    for name in ALLOC_ENV:
        if os.environ.get(name):
            return f"{name}={os.environ[name]}"
    settings = EXPANDABLE if sys.platform == "linux" else ROUNDED
    _SET_ALLOCATOR(settings)
    return settings


def device_budget(device: torch.device | str) -> int:
    """Bytes an `auto` placement may plan for on `device`: free memory now (the context
    and every other process already subtracted), with headroom; the host has no limit.
    """
    if not streamed(device):
        return 1 << 62
    return int(free_bytes(device) * HEADROOM)


def free_bytes(device: torch.device | str) -> int:
    """Free bytes of the accelerator `device` from the driver; raises where the backend
    reports none (an `auto` placement cannot plan there)."""
    info = mem_info(device)
    if info is None:
        raise RuntimeError(
            f"{device}: the backend reports no free memory (mem_get_info)"
        )
    return info[0]
