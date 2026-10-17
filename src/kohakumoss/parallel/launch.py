"""Process launch and process-group init without torchrun: `spawn(fn, plan, *args)`
runs `fn(env, *args)` in one process per device. See
docs/internals/distributed-windows.md."""

import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kohakumoss.types.plans import ParallelPlan

IS_WINDOWS = sys.platform == "win32"


@dataclass(frozen=True)
class DistEnv:
    """This process's place in the job."""

    rank: int
    world: int
    local_rank: int
    device: torch.device
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def default_backend(device_type: str = "cuda") -> str:
    """nccl on Linux CUDA, gloo on Windows / CPU, xccl on XPU when available."""
    if device_type == "cuda" and not IS_WINDOWS and dist.is_nccl_available():
        return "nccl"
    if device_type == "xpu" and dist.is_backend_available("xccl"):
        return "xccl"
    return "gloo"


def _device(device_type: str, index: int) -> torch.device:
    match device_type:
        case "cuda":
            torch.cuda.set_device(index)
            return torch.device("cuda", index)
        case "xpu":
            torch.xpu.set_device(index)
            return torch.device("xpu", index)
        case _:
            return torch.device("cpu")


def init(
    rank: int,
    world: int,
    backend: str,
    init_method: str,
    addr: str,
    port: int,
    file_path: str,
    device: torch.device,
    timeout_s: float = 1800.0,
) -> None:
    """init_process_group via an explicit store (tcp) or a shared file."""
    timeout = timedelta(seconds=timeout_s)
    if init_method == "file":
        url = "file:///" + file_path.replace("\\", "/").lstrip("/")
        dist.init_process_group(
            backend, init_method=url, rank=rank, world_size=world, timeout=timeout
        )
        return
    store = dist.TCPStore(
        addr, port, world, is_master=False, timeout=timeout, use_libuv=not IS_WINDOWS
    )
    dist.init_process_group(
        backend, store=store, rank=rank, world_size=world, timeout=timeout
    )


def _entry(local_rank, fn, cfg, args):
    world = len(cfg["devices"])
    os.environ.update(
        RANK=str(local_rank),
        LOCAL_RANK=str(local_rank),
        WORLD_SIZE=str(world),
        MASTER_ADDR=cfg["addr"],
        MASTER_PORT=str(cfg["port"]),
    )
    if IS_WINDOWS:
        os.environ["USE_LIBUV"] = "0"
    device = _device(cfg["device_type"], cfg["devices"][local_rank])
    init(
        local_rank,
        world,
        cfg["backend"],
        cfg["init"],
        cfg["addr"],
        cfg["port"],
        cfg["file"],
        device,
    )
    env = DistEnv(local_rank, world, local_rank, device, cfg["backend"])
    try:
        fn(env, *args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def spawn(fn, plan: ParallelPlan, *args, device_type: str = "cuda") -> None:
    """Run `fn(env, *args)` on every device of `plan` (one process each); `fn` must be
    a module-level function (spawned processes import it)."""
    backend = default_backend(device_type) if plan.backend == "auto" else plan.backend
    addr = "localhost" if IS_WINDOWS else "127.0.0.1"
    world = len(plan.devices)
    store = dist.TCPStore(
        addr,
        0,
        world,
        is_master=True,
        timeout=timedelta(seconds=1800),
        wait_for_workers=False,
        use_libuv=not IS_WINDOWS,
    )
    cfg = dict(
        devices=tuple(plan.devices),
        device_type=device_type,
        backend=backend,
        init=plan.init,
        addr=addr,
        port=store.port,
        file=os.path.join(tempfile.mkdtemp(prefix="kohakumoss_dist_"), "init"),
    )
    mp.spawn(_entry, args=(fn, cfg, args), nprocs=world, join=True)
    del store


def local_env(device: torch.device | str = "cuda") -> DistEnv:
    """The single-process environment (no process group)."""
    device = torch.device(device)
    return DistEnv(0, 1, 0, device, "none")
