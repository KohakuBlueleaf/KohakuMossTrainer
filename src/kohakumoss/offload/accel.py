"""Device-generic stream / event access for the offload engines: CUDA and XPU run the
side-stream paths, every other device the synchronous host paths."""

import torch

STREAMED = ("cuda", "xpu")


def default_type() -> str:
    """The accelerator a run uses: cuda, else xpu, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def streamed(device: torch.device | str) -> bool:
    """The device has side streams, events and pinned host memory."""
    return torch.device(device).type in STREAMED


def module(device: torch.device | str):
    """`torch.cuda` / `torch.xpu` for `device` (Stream, Event, stream, current_stream)."""
    return torch.get_device_module(torch.device(device))


def with_index(device: torch.device | str) -> torch.device:
    """`device` with its index filled in (the current one when unset)."""
    device = torch.device(device)
    if streamed(device) and device.index is None:
        return torch.device(device.type, module(device).current_device())
    return device
