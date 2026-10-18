"""Process launch and process-group init without torchrun: `spawn(fn, plan, *args)`
runs `fn(env, *args)` in one process per device. See
docs/internals/distributed-launch.md."""

import contextlib
import itertools
import multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kohakumoss.types.plans import ParallelPlan
from kohakumoss_kernels.dist import COLLECTIVE_DEVICE, grouped
from kohakumoss_kernels.dist import dist_rank as dist_rank

IS_WINDOWS = sys.platform == "win32"
TIMEOUT_S = 600.0  # default DIST_TIMEOUT: seconds a collective / the store waits
RANK0_WAIT_S = 7200.0  # default DIST_GATE_TIMEOUT: seconds a store gate waits
EXIT_WAIT_S = 30.0  # seconds a terminated rank process gets before it is killed
DEVICE_BACKENDS = tuple(COLLECTIVE_DEVICE)
BINDS = ("lazy", "device")
_GATES = itertools.count()
_STARTUP = {"timeout_s": RANK0_WAIT_S}


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
    timeout_s: float = TIMEOUT_S,
    kind: str = "ddp",
    bind: str = "lazy",
) -> None:
    """init_process_group via an explicit store (tcp) or a shared file; every
    collective and the store wait `timeout_s`; `bind` as in `bound_device`."""
    timeout = timedelta(seconds=timeout_s)
    bound = bound_device(backend, device, kind, bind)
    if init_method == "file":
        url = "file:///" + file_path.replace("\\", "/").lstrip("/")
        dist.init_process_group(
            backend,
            init_method=url,
            rank=rank,
            world_size=world,
            timeout=timeout,
            **bound,
        )
        return
    store = dist.TCPStore(
        addr, port, world, is_master=False, timeout=timeout, use_libuv=not IS_WINDOWS
    )
    dist.init_process_group(
        backend, store=store, rank=rank, world_size=world, timeout=timeout, **bound
    )


def bound_device(
    backend: str, device: torch.device, kind: str = "ddp", bind: str = "lazy"
) -> dict:
    """`{"device_id": device}` with DIST_BIND='device' for a device backend (nccl /
    xccl) on its accelerator outside the pipeline; else nothing (lazy init). See
    docs/internals/distributed-launch.md."""
    if bind not in BINDS:
        raise ValueError(f"DIST_BIND={bind!r} not in {BINDS}")
    device_backend = backend in DEVICE_BACKENDS and device.type != "cpu"
    if bind == "device" and device_backend and kind != "pipeline":
        return {"device_id": device}
    return {}


def _store(store=None):
    return store if store is not None else dist.distributed_c10d._get_default_store()


def release_ranks(name: str, store=None, payload: str = "done") -> None:
    """Rank 0: its one-time work `name` is done (`wait_for_rank0` returns `payload`)."""
    _store(store).set(f"kohakumoss/{name}", payload)


def set_gate_timeout(seconds: float) -> None:
    """DIST_GATE_TIMEOUT: seconds the store gates (`wait_for_rank0`, `rank0_value`,
    `startup_barrier` / `ranks_meet`: startup, checkpoint saves, the sample grid) wait,
    for this process."""
    if not seconds > 0:
        raise ValueError(f"gate timeout must be > 0 seconds, got {seconds!r}")
    _STARTUP["timeout_s"] = float(seconds)


def gate_timeout() -> float:
    """Seconds a store gate waits (DIST_GATE_TIMEOUT, default RANK0_WAIT_S)."""
    return _STARTUP["timeout_s"]


def _wait(store, key: str, timeout_s: float | None) -> None:
    seconds = gate_timeout() if timeout_s is None else timeout_s
    store.wait([key], timedelta(seconds=seconds))


def wait_for_rank0(name: str, store=None, timeout_s: float | None = None) -> str:
    """The other ranks: wait on the store, not in a collective, until rank 0 calls
    `release_ranks(name)`, for up to `timeout_s` (None: the gate timeout); the
    payload rank 0 released with."""
    s, key = _store(store), f"kohakumoss/{name}"
    _wait(s, key, timeout_s)
    return s.get(key).decode()


def rank0_value(
    name: str,
    choose: Callable[[], str],
    rank: int,
    world: int,
    store=None,
    timeout_s: float | None = None,
) -> str:
    """`choose()` run on rank 0 only; every rank returns its result (rank 0 releases
    gate `name` with it, the others wait up to `timeout_s`; one rank: `choose()`)."""
    if world == 1:
        return choose()
    if rank == 0:
        value = choose()
        release_ranks(name, store, value)
        return value
    return wait_for_rank0(name, store, timeout_s)


def startup_barrier(
    name: str, world: int, store=None, timeout_s: float | None = None
) -> None:
    """Every rank: wait on the store, not in a collective, until all `world` ranks
    reached `name`, for up to `timeout_s` (None: the gate timeout)."""
    s = _store(store)
    key = f"kohakumoss/{name}"
    if s.add(f"{key}/arrived", 1) == world:
        s.set(key, "done")
    _wait(s, key, timeout_s)


def ranks_meet(what: str) -> None:
    """`startup_barrier` over the default process group (nothing without one or with
    one rank); a distinct gate per call."""
    if grouped():
        startup_barrier(gate_name(what), dist.get_world_size())


def gate_name(what: str) -> str:
    """A name for one rank-0-first gate, distinct per call in the run."""
    return f"{what}-{next(_GATES)}"


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
        timeout_s=cfg["timeout_s"],
        kind=cfg["kind"],
        bind=cfg["bind"],
    )
    env = DistEnv(local_rank, world, local_rank, device, cfg["backend"])
    try:
        fn(env, *args)
    except BaseException:
        # report, kill the workers, exit; see docs/internals/distributed-launch.md
        traceback.print_exc()
        sys.stderr.flush()
        for child in multiprocessing.active_children():
            child.kill()
        os._exit(1)
    if dist.is_initialized():
        dist.destroy_process_group()


def spawn(fn, plan: ParallelPlan, *args, device_type: str = "cuda") -> None:
    """Run `fn(env, *args)` on every device of `plan` (one process each); `fn` must be
    a module-level function (spawned processes import it). The `init="file"`
    rendezvous directory is removed when the job ends: on a normal exit, a failing
    rank, or SIGTERM to the launcher (which also ends the rank processes)."""
    backend = default_backend(device_type) if plan.backend == "auto" else plan.backend
    addr = "localhost" if IS_WINDOWS else "127.0.0.1"
    world = len(plan.devices)
    store = dist.TCPStore(
        addr,
        0,
        world,
        is_master=True,
        timeout=timedelta(seconds=plan.timeout_s),
        wait_for_workers=False,
        use_libuv=not IS_WINDOWS,
    )
    rendezvous = (
        tempfile.mkdtemp(prefix="kohakumoss_dist_") if plan.init == "file" else ""
    )
    cfg = dict(
        devices=tuple(plan.devices),
        device_type=device_type,
        backend=backend,
        init=plan.init,
        kind=plan.kind,
        timeout_s=plan.timeout_s,
        bind=plan.bind,
        addr=addr,
        port=store.port,
        file=os.path.join(rendezvous, "init") if rendezvous else "",
    )
    try:
        with _sigterm_exits():
            _run(fn, cfg, args, world)
    finally:
        if rendezvous:
            shutil.rmtree(rendezvous, ignore_errors=True)
        del store


def _run(fn, cfg: dict, args: tuple, world: int) -> None:
    """The rank processes, joined; any still running when the join ends early (a
    SIGTERM) are terminated, then killed after EXIT_WAIT_S."""
    context = mp.spawn(_entry, args=(fn, cfg, args), nprocs=world, join=False)
    try:
        while not context.join():
            pass
    finally:
        alive = [p for p in context.processes if p.is_alive()]
        for proc in alive:
            proc.terminate()
        for proc in alive:
            proc.join(EXIT_WAIT_S)
            if proc.is_alive():
                proc.kill()


def _exit_on_signal(signum, frame) -> None:
    raise SystemExit(128 + signum)


@contextlib.contextmanager
def _sigterm_exits():
    """Inside, SIGTERM to the (main-thread) launcher raises SystemExit so the job's
    clean-up runs; the default handler is restored after. Off the main thread, or with
    a SIGTERM handler already set, nothing changes."""
    main = threading.current_thread() is threading.main_thread()
    if not main or signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
        yield
        return
    signal.signal(signal.SIGTERM, _exit_on_signal)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
