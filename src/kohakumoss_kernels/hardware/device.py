"""Device description the planners rank configs with: driver-queried fields plus
measured rates (16-bit MMA with fp32 accumulation, DRAM and L2 bandwidth), cached
per card on disk so each card is probed once. See docs/internals/tuning.md.
"""

import dataclasses
import math
import os
from functools import cache
from pathlib import Path

import torch
from torch._guards import detect_fake_mode

from kohakumoss_common.io import read_json, write_json_atomic
from kohakumoss_common.log import get_logger
from kohakumoss_kernels.hardware.arch import Arch, _cuda_arch, arch_of
from kohakumoss_kernels.hardware.current import rng_kept, synchronize, with_index

_UNWRITABLE: set[str] = set()
log = get_logger(__name__)


def cache_dir() -> Path:
    """Root of every on-disk kernel cache (`KOHAKUMOSS_KERNEL_CACHE` overrides)."""
    root = os.environ.get("KOHAKUMOSS_KERNEL_CACHE")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "kohakumoss_kernels"


def write_json(path: Path, data: dict) -> None:
    """Atomic write (`kohakumoss_common.io.write_json_atomic`): a reader never sees a
    half-written table. Raises OSError."""
    write_json_atomic(path, data, indent=1, sort_keys=True)


def save_json(path: Path, data: dict) -> bool:
    """`write_json`, an OSError logged at WARNING (once per path) instead of raised;
    whether the file was written."""
    try:
        write_json(path, data)
        return True
    except OSError as err:
        if str(path) not in _UNWRITABLE:
            _UNWRITABLE.add(str(path))
            log.warning(
                f"cannot write {path} ({type(err).__name__}: {err}); the values stay "
                "in memory for this process"
            )
        return False


@dataclasses.dataclass(frozen=True)
class Device:
    """One card: queried limits and measured rates (TFLOP/s, GB/s)."""

    name: str
    kind: str
    sm: int
    sms: int
    smem_per_block: int
    smem_per_sm: int
    regs_per_sm: int
    l2_bytes: int
    memory_bytes: int
    mma_tflops: float = math.nan
    dram_gbs: float = math.nan
    l2_gbs: float = math.nan

    @property
    def arch(self) -> Arch:
        if self.kind == "cuda":
            return _cuda_arch(self.sm // 10, self.sm % 10)
        return arch_of(self.kind)

    @property
    def key(self) -> str:
        """Cache key: card, arch, CUDA and torch versions."""
        cuda = torch.version.cuda or torch.version.hip or "none"
        name = self.name.replace(" ", "_").replace("/", "_")
        return f"{name}-sm{self.sm}-cu{cuda}-torch{torch.__version__.split('+')[0]}"

    @property
    def measured(self) -> bool:
        return not any(
            math.isnan(v) for v in (self.mma_tflops, self.dram_gbs, self.l2_gbs)
        )


def query(device: torch.device | str) -> Device:
    """Driver-reported fields of `device` (measured rates unset)."""
    device = torch.device(device)
    match device.type:
        case "cuda":
            p = torch.cuda.get_device_properties(device)
            return Device(
                name=p.name,
                kind="cuda",
                sm=p.major * 10 + p.minor,
                sms=p.multi_processor_count,
                smem_per_block=getattr(p, "shared_memory_per_block_optin", 49152),
                smem_per_sm=getattr(p, "shared_memory_per_multiprocessor", 65536),
                regs_per_sm=getattr(p, "regs_per_multiprocessor", 65536),
                l2_bytes=getattr(p, "L2_cache_size", 4 << 20),
                memory_bytes=p.total_memory,
            )
        case "xpu":
            p = torch.xpu.get_device_properties(device)
            return Device(
                name=p.name,
                kind="xpu",
                sm=0,
                sms=getattr(p, "gpu_subslice_count", 16),
                smem_per_block=getattr(p, "local_mem_size", 65536),
                smem_per_sm=getattr(p, "local_mem_size", 65536),
                regs_per_sm=65536,
                l2_bytes=4 << 20,
                memory_bytes=p.total_memory,
            )
        case _:
            return Device("cpu", "cpu", 0, os.cpu_count() or 1, 0, 0, 0, 0, 0)


def _rate(fn, device: torch.device, reps: int) -> float:
    """Best seconds per call over a few rounds of `reps` calls."""
    fn()
    synchronize(device)
    best = math.inf
    for _ in range(3):
        start = torch.Event(device=device, enable_timing=True)
        end = torch.Event(device=device, enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        synchronize(device)
        best = min(best, start.elapsed_time(end) / 1e3 / reps)
    return best


def measure(dev: Device, device: torch.device) -> Device:
    """Fill the measured rates with three short microbenchmarks."""
    dtype = torch.float16
    n = 4096
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    mma = 2 * n**3 / _rate(lambda: torch.mm(a, b), device, 10) / 1e12
    big = torch.empty(
        max(dev.l2_bytes * 16, 256 << 20) // 2, device=device, dtype=dtype
    )
    dst = torch.empty_like(big)
    dram = 2 * big.numel() * 2 / _rate(lambda: dst.copy_(big), device, 5) / 1e9
    small = big[: max(dev.l2_bytes // 8, 1 << 16)]
    sdst = dst[: small.numel()]
    l2 = 2 * small.numel() * 2 / _rate(lambda: sdst.copy_(small), device, 200) / 1e9
    return dataclasses.replace(dev, mma_tflops=mma, dram_gbs=dram, l2_gbs=l2)


def tracing() -> bool:
    """Inside torch.compile / fake-tensor tracing / CUDA graph capture: no measuring."""
    if torch.compiler.is_compiling() or detect_fake_mode() is not None:
        return True
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def planning_device(device: torch.device | str) -> Device:
    """`measure_rates(device)` when real work may run, else the unmeasured
    `query(device)`."""
    return query(device) if tracing() else measure_rates(device)


@cache
def measure_rates(device: torch.device | str) -> Device:
    """The measured description of `device`, from the on-disk cache when present."""
    device = with_index(device)
    dev = query(device)
    if dev.kind == "cpu":
        return dev
    path = cache_dir() / "devices" / f"{dev.key}.json"
    data = read_json(path)
    if data is not None:
        names = {f.name for f in dataclasses.fields(Device)}
        return Device(**{k: v for k, v in data.items() if k in names})
    with rng_kept([device]):
        dev = measure(dev, device)
    save_json(path, dataclasses.asdict(dev))
    return dev
