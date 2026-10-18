"""The device's memory use from the driver (CUDA / XPU `mem_get_info`: total - free,
what nvidia-smi shows) and its breakdown: other processes, this process's context,
non-torch allocations, the caching allocator's free cache and fragmentation, allocated
tensors. Other processes and the context are measured once, at construction. See
docs/guides/logging.md#vram."""

import torch

from kohakumoss.logging.log import log_table

GIB = 2**30
PARTS = ("other", "context", "non_torch", "fragmented", "cached", "allocated")
WHAT = {
    "other": "other processes and the driver, before this process's context",
    "context": "CUDA / XPU context (driver, runtime)",
    "non_torch": "outside the torch allocator: kernel modules, library state, NCCL",
    "fragmented": "torch reserved, free inside split blocks",
    "cached": "torch reserved, free whole blocks",
    "allocated": "live tensors",
}


def _mem(device: torch.device):
    return torch.cuda if device.type == "cuda" else torch.xpu


def _nvml_used(device: torch.device) -> int | None:
    """Device-wide used bytes from NVML / SMI without creating a context, else None."""
    used = getattr(torch.cuda, "device_memory_used", None)
    if device.type != "cuda" or used is None:
        return None
    try:
        return int(used(device))
    except Exception:  # noqa: BLE001 - no NVML / pynvml: the context is not separated
        return None


class DeviceUse:
    """Construct before the process's first allocation on `device`. Without NVML the
    other processes are not separated from the context (`other` is 0, `context` holds
    both). `mem` (torch.cuda / torch.xpu) and `nvml_used` are injectable."""

    def __init__(self, device, mem=None, nvml_used=_nvml_used):
        self.device = torch.device(device)
        self.mem = mem if mem is not None else _mem(self.device)
        before = nvml_used(self.device)
        after = self.used()
        self.separated = before is not None
        self.other = before if self.separated else 0
        self.context = max(after - self.other, 0)

    def used(self) -> int:
        """The device's used bytes now, from the driver (total - free; every process,
        this one's context included)."""
        free, total = self.mem.mem_get_info(self.device)
        return total - free

    def breakdown(self) -> dict[str, int]:
        """Bytes per part (PARTS) plus `device`; the parts sum to `device`."""
        use = self.used()
        reserved = self.mem.memory_reserved(self.device)
        allocated = self.mem.memory_allocated(self.device)
        stats = self.mem.memory_stats(self.device)
        split = int(stats.get("inactive_split_bytes.all.current", 0))
        free = reserved - allocated
        fragmented = min(split, free)
        return {
            "other": self.other,
            "context": self.context,
            "non_torch": use - self.other - self.context - reserved,
            "fragmented": fragmented,
            "cached": free - fragmented,
            "allocated": allocated,
            "device": use,
        }

    def metrics(self) -> dict[str, float]:
        """`vram_<part>` GiB for the metric rows."""
        return {f"vram_{k}": v / GIB for k, v in self.breakdown().items()}

    def log(self, log, when: str) -> dict[str, int]:
        """One table of the breakdown under `vram <when>`; returns the bytes."""
        parts = self.breakdown()
        rows = [(p, round(parts[p] / GIB, 2), WHAT[p]) for p in PARTS]
        total = f"{parts['device'] / GIB:.2f} GiB on the device"
        other = "measured" if self.separated else "in context"
        title = f"vram {when} on {self.device}: {total} (other processes {other})"
        log_table(log, title, ("part", "GiB", "what"), rows)
        return parts


class NoDeviceUse:
    """The host: no device accounting."""

    def used(self) -> int:
        return 0

    def breakdown(self) -> dict[str, int]:
        return {}

    def metrics(self) -> dict[str, float]:
        return {}

    def log(self, log, when: str) -> dict[str, int]:
        return {}


def build_device_use(device):
    device = torch.device(device)
    if device.type in ("cuda", "xpu") and hasattr(_mem(device), "mem_get_info"):
        return DeviceUse(device)
    return NoDeviceUse()
