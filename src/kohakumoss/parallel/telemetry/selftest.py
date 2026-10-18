"""Multi-process checks and link measurements, model-free (importable by spawned
workers): the gradient reducer against the sum it must produce, and all-reduce /
all-gather / reduce-scatter / point-to-point bandwidth. Each worker writes its result
to `<path>.<rank>.json`."""

import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from functools import partial

import torch
import torch.distributed as dist
from torch import nn

from kohakumoss.offload.optim.cpu_optim import adamw_host
from kohakumoss.offload.weights.shared import (
    SharedHost,
    aligned,
    block_owners,
    nbytes_of,
    owned,
)
from kohakumoss.parallel.ddp import AsyncReducer, GradReducer
from kohakumoss.parallel.launch import spawn
from kohakumoss_common.units import MIB
from kohakumoss_kernels.hardware.current import synchronize

SHAPES = ((3, 5), (7,), (2, 4, 6), (1,), (33, 9))


def _write(path: str, rank: int, payload: dict) -> None:
    with open(f"{path}.{rank}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _coefficients(rank: int, micro: int, device) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(1000 * rank + micro)
    return [torch.randn(s, generator=g).to(device) for s in SHAPES]


def ddp_worker(
    env,
    result_path: str,
    micro: int = 2,
    bucket_bytes: int = 256,
    average: bool = False,
) -> None:
    """Every rank backprops `sum(p * c[rank, m])` over `micro` micro-batches through a
    GradReducer with small buckets; each gradient must equal the sum of `c` over every
    rank and micro-batch (computed locally from the seeds). `average` builds the
    reducer averaging instead (a check of the sum must fail on it)."""
    params = [nn.Parameter(torch.zeros(s, device=env.device)) for s in SHAPES]
    reducer = GradReducer(params, bucket_bytes=bucket_bytes, average=average)
    for m in range(micro):
        reducer.start(m == micro - 1)
        coef = _coefficients(env.rank, m, env.device)
        sum((p * c).sum() for p, c in zip(params, coef)).backward()
    reducer.finish()
    want = [torch.zeros(s, device=env.device) for s in SHAPES]
    for r in range(env.world):
        for m in range(micro):
            for w, c in zip(want, _coefficients(r, m, env.device)):
                w += c
    err = max((p.grad - w).abs().max().item() for p, w in zip(params, want))
    views = all(
        p.grad.data_ptr() == v.data_ptr()
        for b in reducer.buckets
        for p, v in zip(b.params, b.views)
    )
    _write(result_path, env.rank, {"max_err": err, "views": views})


class _LastHook:
    """Post-accumulate hook body active only on the last micro-batch."""

    def __init__(self, body):
        self.body = body
        self.on = False

    def __call__(self, p: torch.Tensor) -> None:
        if self.on:
            self.body(p)
            p.grad = None


def _record(got: dict, p: torch.Tensor) -> None:
    got[id(p)] = p.grad.clone()
    p.grad = None


def _local(flat: torch.Tensor) -> None:
    return None


def _delivered_err(got: dict, p: torch.Tensor, want: torch.Tensor) -> float:
    """Max abs error of `p`'s delivered gradient; inf when none was delivered."""
    if id(p) not in got:
        return float("inf")
    return (got[id(p)] - want).abs().max().item()


def async_worker(
    env,
    result_path: str,
    micro: int = 2,
    bucket_bytes: int = 256,
    broken: bool = False,
) -> None:
    """Every rank backprops `sum(p * c[rank, m])` over `micro` micro-batches; on the last
    one each accumulated gradient leaves through an AsyncReducer (small buckets) from
    its post-accumulate hook, and the delivered gradient must equal the sum of `c` over
    every rank and micro-batch. Parameter 1 has no gradient on rank 1 (it counts as
    zero there, and is delivered there too). A flat buffer summed by `reduce_flat` on
    the caller's stream must hold the sum of the ranks' values. `broken` skips the
    reduction (a check of the sum must fail on it; an undelivered gradient is an
    infinite error)."""
    params = [nn.Parameter(torch.zeros(s, device=env.device)) for s in SHAPES]
    reducer = AsyncReducer(env.device, bucket_bytes=bucket_bytes)
    if broken:
        reducer.reduce_flat = _local
    got: dict[int, torch.Tensor] = {}
    hook = _LastHook(reducer.reduced(params, partial(_record, got)))
    for p in params:
        p.register_post_accumulate_grad_hook(hook)
    skip = 1 if env.rank == 1 else None
    for m in range(micro):
        last = m == micro - 1
        reducer.start(last)
        hook.on = last
        coef = _coefficients(env.rank, m, env.device)
        terms = [
            (p * c).sum() for i, (p, c) in enumerate(zip(params, coef)) if i != skip
        ]
        sum(terms).backward()
    reducer.finish()
    flat = torch.full((7,), float(env.rank + 1), device=env.device)
    reducer.reduce_flat(flat)
    synchronize(env.device)
    want = [torch.zeros(s, device=env.device) for s in SHAPES]
    for r in range(env.world):
        for m in range(micro):
            for i, (w, c) in enumerate(zip(want, _coefficients(r, m, env.device))):
                if not (r == 1 and i == 1):
                    w += c
    err = max(_delivered_err(got, p, w) for p, w in zip(params, want))
    flat_want = env.world * (env.world + 1) / 2
    flat_err = (flat - flat_want).abs().max().item()
    shapes = all(id(p) in got and got[id(p)].shape == p.shape for p in params)
    payload = {
        "max_err": max(err, flat_err),
        "views": shapes,
        "buckets": len(reducer.slabs),
    }
    _write(result_path, env.rank, payload)


HYPER = (1e-2, 0.9, 0.999, 1e-8, 0.01)  # lr, beta1, beta2, eps, weight decay


def _block_grad(rank: int, step: int, block: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(10_000 * step + 100 * block + rank)
    return torch.randn(SHAPES[block], generator=g)


def _skip_reduce(flat: torch.Tensor, index: int) -> None:
    return None


def zero_worker(env, result_path: str, steps: int = 2, broken: bool = False) -> None:
    """Host optimiser shards over shared masters (CPU, gloo): the blocks of SHAPES live
    once in a SharedHost region; each step every block's gradient is summed onto its
    owner (`AsyncReducer.block_reduce`), the owner alone runs the host AdamW with its
    own moments and the ranks meet at `ready()`. On every rank the shared masters must
    equal the reference (summed gradient, every rank stepping a private copy with full
    moments). `broken` skips the reduction (a check of the result must fail on it)."""
    owners = block_owners(len(SHAPES), env.world)
    size = sum(aligned(nbytes_of(s, torch.float32)) for s in SHAPES)
    arena = SharedHost(size)
    shared = [arena.take(s, torch.float32) for s in SHAPES]
    init = [torch.randn(s, generator=torch.Generator().manual_seed(7)) for s in SHAPES]
    if arena.writer:
        for t, v in zip(shared, init):
            t.copy_(v)
    arena.ready()
    mine = owned(owners, env.rank)
    state = {b: [torch.zeros(SHAPES[b]) for _ in range(2)] for b in mine}
    steps_of = {b: torch.zeros(()) for b in mine}
    ref = [v.clone() for v in init]
    ref_state = [[torch.zeros(s) for _ in range(2)] for s in SHAPES]
    ref_steps = [torch.zeros(()) for _ in SHAPES]
    reduce = AsyncReducer(env.device).block_reduce(owners)
    if broken:
        reduce = _skip_reduce
    for step in range(steps):
        for b in range(len(SHAPES)):
            flat = _block_grad(env.rank, step, b).reshape(-1)
            reduce(flat, b)
            if b in state:
                m, v = state[b]
                adamw_host(shared[b], flat.view(SHAPES[b]), m, v, steps_of[b], *HYPER)
            full = sum(_block_grad(r, step, b) for r in range(env.world))
            m, v = ref_state[b]
            adamw_host(ref[b], full, m, v, ref_steps[b], *HYPER)
        arena.ready()
    err = max((s - r).abs().max().item() for s, r in zip(shared, ref))
    payload = {"max_err": err, "owned": mine, "blocks": len(SHAPES)}
    _write(result_path, env.rank, payload)
    arena.ready()


FAILURE = "constructed rank failure"
RAISED = "selftest rank raised at"


def _raised() -> None:
    """One stderr line with the wall time the failing rank raises at (epoch s)."""
    print(f"{RAISED} {time.time():.3f}", file=sys.stderr, flush=True)


def raise_worker(env, teardown: bool = False) -> None:
    """Rank 1 posts a receive from rank 0 and raises while rank 0 waits in a receive
    from it: the launcher must end the whole job at once. `teardown` makes the failing
    rank destroy its process group before raising (on nccl that blocks on the posted
    receive; the job must then hang). The raise time goes to stderr (`RAISED`)."""
    t = torch.zeros(1, device=env.device)
    if env.rank == 0:
        dist.recv(t, src=1)
        return
    dist.irecv(t, src=0)
    try:
        _raised()
        raise RuntimeError(FAILURE)
    except RuntimeError:
        if teardown:
            dist.destroy_process_group()
        raise


def _linger(pid_path: str) -> None:
    with open(pid_path + ".tmp", "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    os.replace(pid_path + ".tmp", pid_path)
    time.sleep(600)


def orphan_worker(env, pid_path: str, detach: bool = False) -> None:
    """Rank 1 starts a lingering forkserver child (a loader worker's shape; its pid in
    `pid_path`) and raises while rank 0 waits in a receive from it: the launcher must
    end the job and the child. `detach` hides the child from the rank's process table
    (a launcher that leaves children alive; the child must then survive)."""
    t = torch.zeros(1, device=env.device)
    if env.rank == 0:
        dist.recv(t, src=1)
        return
    child = mp.get_context("forkserver").Process(target=_linger, args=(pid_path,))
    child.start()
    while not os.path.exists(pid_path):
        time.sleep(0.05)
    if detach:
        mp.process._children.discard(child)
    _raised()
    raise RuntimeError(FAILURE)


def _timed(fn, device, iters: int = 10) -> float:
    for _ in range(3):
        fn()
    synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - t0) / iters


def comm_worker(env, result_path: str, sizes_mb: tuple[int, ...] = (1, 16, 64)) -> None:
    """all_reduce time and algorithm GB/s per message size."""
    out = {}
    for mb in sizes_mb:
        t = torch.ones(mb * MIB // 4, device=env.device)
        dt = _timed(lambda: dist.all_reduce(t), env.device)
        out[str(mb)] = {"ms": dt * 1e3, "algo_GBps": mb / 1024 / dt}
    _write(result_path, env.rank, out)


def collective_worker(
    env, result_path: str, sizes_mb: tuple[int, ...] = (16, 64, 256)
) -> None:
    """all_gather_into_tensor / reduce_scatter_tensor (FSDP's collectives) time and
    algorithm GB/s of the full tensor per size (bf16)."""
    out = {}
    for mb in sizes_mb:
        n = mb * MIB // 2 // env.world * env.world
        full = torch.ones(n, dtype=torch.bfloat16, device=env.device)
        part = torch.ones(n // env.world, dtype=torch.bfloat16, device=env.device)
        gather = _timed(lambda: dist.all_gather_into_tensor(full, part), env.device)
        scatter = _timed(lambda: dist.reduce_scatter_tensor(part, full), env.device)
        out[str(mb)] = {
            "all_gather_ms": gather * 1e3,
            "all_gather_GBps": mb / 1024 / gather,
            "reduce_scatter_ms": scatter * 1e3,
            "reduce_scatter_GBps": mb / 1024 / scatter,
        }
    _write(result_path, env.rank, out)


def hop_worker(
    env, result_path: str, sizes_kb: tuple[int, ...] = (4, 96, 1024, 16384, 65536)
) -> None:
    """Point-to-point hop time and GB/s per size from full circulations of the rank
    ring (rank 0 sends first; every other rank receives, then sends on)."""
    nxt, prv = (env.rank + 1) % env.world, (env.rank - 1) % env.world
    out = {}
    for kb in sizes_kb:
        buf = torch.empty(kb * 1024, dtype=torch.uint8, device=env.device)

        def ring():
            if env.rank == 0:
                dist.send(buf, nxt)
                dist.recv(buf, prv)
            else:
                dist.recv(buf, prv)
                dist.send(buf, nxt)

        hop = _timed(ring, env.device, iters=20) / env.world
        out[str(kb)] = {"hop_ms": hop * 1e3, "GBps": kb / 2**20 / hop}
    _write(result_path, env.rank, out)


def read_results(path: str, world: int) -> list[dict]:
    out = []
    for r in range(world):
        with open(f"{path}.{r}.json", encoding="utf-8") as f:
            out.append(json.load(f))
        os.remove(f"{path}.{r}.json")
    return out


def spawn_results(worker, plan, *args, device_type: str = "cuda") -> list[dict]:
    """Every rank's results of `worker(env, path, *args)` run through the launcher on
    `plan`'s devices; the result files live in a temporary directory removed
    afterwards."""
    with tempfile.TemporaryDirectory(prefix="kohakumoss_results_") as root:
        path = os.path.join(root, "res")
        spawn(worker, plan, path, *args, device_type=device_type)
        return read_results(path, len(plan.devices))


__all__ = [
    "async_worker",
    "collective_worker",
    "comm_worker",
    "ddp_worker",
    "hop_worker",
    "orphan_worker",
    "raise_worker",
    "read_results",
    "spawn_results",
    "zero_worker",
]
