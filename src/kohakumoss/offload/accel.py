"""Device-generic stream / event access for the offload engines: CUDA and XPU run the
side-stream paths, every other device the synchronous host paths."""

import os
import sys

import torch

from kohakumoss_common.devices import streamed_type

EXPANDABLE = "expandable_segments:True"
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


def configure_allocator(device: torch.device | str) -> str:
    """Before the run's first allocation: expandable segments for the CUDA caching
    allocator where it has them (Linux), torch's defaults elsewhere; a
    PYTORCH_CUDA_ALLOC_CONF / PYTORCH_ALLOC_CONF in the environment wins. Returns the
    settings in force. See docs/internals/offload.md#allocator."""
    device = torch.device(device)
    if device.type != "cuda":
        return f"{device.type} default"
    for name in ALLOC_ENV:
        if os.environ.get(name):
            return f"{name}={os.environ[name]}"
    if sys.platform != "linux":
        return "default"
    _SET_ALLOCATOR(EXPANDABLE)
    return EXPANDABLE


def with_index(device: torch.device | str) -> torch.device:
    """`device` with its index filled in (the current one when unset)."""
    device = torch.device(device)
    if streamed(device) and device.index is None:
        return torch.device(device.type, module(device).current_device())
    return device


def device_budget(device: torch.device | str) -> int:
    """Bytes an `auto` placement may plan for on `device`: free memory now (the context
    and every other process already subtracted), with headroom; the host has no limit.
    """
    device = torch.device(device)
    match device.type:
        case "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            return int(free * HEADROOM)
        case "xpu":
            free, _ = torch.xpu.mem_get_info(device)
            return int(free * HEADROOM)
        case _:
            return 1 << 62
