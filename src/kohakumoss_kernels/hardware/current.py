"""The accelerator device types and their torch modules, the current-device guard for
kernel launches, device synchronisation, random draws that keep the caller's
generators (`rng_kept`), and whether saved-tensor hooks are active. See
docs/internals/kernel-dev.md#current-device."""

from collections.abc import Iterable
from contextlib import contextmanager, nullcontext

import torch

from kohakumoss_common.devices import STREAMED

ACCEL = STREAMED
_MODULES = {kind: getattr(torch, kind) for kind in ACCEL}


def is_accel(device: torch.device | str) -> bool:
    """Whether `device` is a CUDA or XPU device."""
    return torch.device(device).type in _MODULES


def device_module(device: torch.device | str):
    """torch.cuda / torch.xpu for an accelerator `device`, None for the host."""
    return _MODULES.get(torch.device(device).type)


def with_index(device: torch.device | str) -> torch.device:
    """`device` with its index filled in (an accelerator's current one when unset; the
    host as given)."""
    device = torch.device(device)
    mod = _MODULES.get(device.type)
    if mod is not None and device.index is None:
        return torch.device(device.type, mod.current_device())
    return device


def on_device(device: torch.device):
    """Context making `device` current; a no-op when it already is."""
    mod = _MODULES.get(device.type)
    if mod is None or device.index is None or device.index == mod.current_device():
        return nullcontext()
    return mod.device(device.index)


def synchronize(device: torch.device) -> None:
    """Wait for every queued kernel on `device` (a no-op on the host)."""
    mod = _MODULES.get(device.type)
    if mod is not None:
        mod.synchronize(device)


@contextmanager
def rng_kept(devices: Iterable[torch.device | str] = ()):
    """Random draws inside (tuning and prewarm inputs) leave the caller's CPU generator
    and the generators of the accelerator `devices` (one device type) where they were,
    so a run's random stream never depends on what the kernel cache held."""
    accel = [d for d in map(torch.device, devices) if d.type in _MODULES]
    kind = accel[0].type if accel else "cuda"
    mod = _MODULES[kind]
    if not mod.is_available():
        accel = []
    index = sorted(
        {mod.current_device() if d.index is None else d.index for d in accel}
    )
    with torch.random.fork_rng(devices=index, device_type=kind):
        yield


def saved_tensor_hooks_active() -> bool:
    """Whether saved-tensor hooks are installed (a checkpointed forward or its
    recompute, activation offload)."""
    return torch._C._autograd._top_saved_tensors_default_hooks(False) is not None
