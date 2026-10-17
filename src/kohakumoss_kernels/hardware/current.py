"""The current-device guard for kernel launches (Triton and TileLang launch on the
current device's queue). See docs/internals/kernel-dev.md."""

from contextlib import nullcontext

import torch

_CURRENT = {"cuda": torch.cuda.current_device, "xpu": torch.xpu.current_device}
_GUARD = {"cuda": torch.cuda.device, "xpu": torch.xpu.device}


def on_device(device: torch.device):
    """Context making `device` current; a no-op when it already is."""
    current = _CURRENT.get(device.type)
    if current is None or device.index is None or device.index == current():
        return nullcontext()
    return _GUARD[device.type](device.index)
