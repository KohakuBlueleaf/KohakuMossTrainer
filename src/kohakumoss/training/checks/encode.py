"""Knob checks of the encode feed (ENCODE_DEVICES, ENCODE_DEPTH, ENCODE_CHUNK)."""

import torch

from kohakumoss.offload.accel import default_type
from kohakumoss.training.checks.encoders import text_problems
from kohakumoss_kernels.hardware.arch import arch_of


def _count(kind: str) -> int:
    match kind:
        case "cuda":
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        case "xpu":
            return torch.xpu.device_count() if torch.xpu.is_available() else 0
        case _:
            return 0


def _positive_int(k: dict, name: str) -> list[str]:
    value = k.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return [f"{name} must be an int >= 1, got {value!r}"]
    return []


def encode_problems(k: dict, arch=arch_of) -> list[str]:
    """ENCODE_DEVICES: accelerator indices, none of them a training device, TEXT_DEVICE
    unset; the text-encoder plan checked on each; ENCODE_DEPTH / ENCODE_CHUNK >= 1."""
    devices = k.get("ENCODE_DEVICES") or ()
    if not devices:
        return []
    if not isinstance(devices, (list, tuple)) or not all(
        isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in devices
    ):
        return [f"ENCODE_DEVICES must be device indices, got {devices!r}"]
    p = _positive_int(k, "ENCODE_DEPTH") + _positive_int(k, "ENCODE_CHUNK")
    shared = sorted(set(devices) & set(k.get("DEVICES") or ()))
    if shared:
        p.append(f"ENCODE_DEVICES {shared} are also training DEVICES")
    if len(set(devices)) != len(devices):
        p.append(f"ENCODE_DEVICES repeats a device: {devices!r}")
    if k.get("TEXT_DEVICE"):
        p.append("TEXT_DEVICE must be '' with ENCODE_DEVICES (the feed places it)")
    kind = default_type()
    if kind == "cpu":
        return p + ["ENCODE_DEVICES needs CUDA or XPU devices"]
    count = _count(kind)
    missing = [i for i in devices if i >= count]
    if missing:
        p.append(f"ENCODE_DEVICES {missing} not present ({count} {kind} devices)")
    for i in devices:
        if i < count:
            p += text_problems({**k, "TEXT_DEVICE": f"{kind}:{i}"}, arch)
    return p
