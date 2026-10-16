"""Shared pieces of the TileLang block ops: compile cache, segment tile tables.

Kernels take caller-allocated outputs (TileLang-allocated outputs lose autograd when a
custom op returns them) and a symbolic token count, so one compile serves every N.
Segment tile tables (sample, first row, end row per tile) are built with device torch
ops from `seg` (no host sync) and cached per `seg` tensor for the rest of the step.
"""

import threading
import weakref
from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch
from torch import Tensor

from kohakumoss_kernels.hardware.device import probe
from kohakumoss_kernels.tuning.tuner import select

DTYPE = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}

_CACHE: dict[tuple, object] = {}
_LOCK = threading.Lock()


class CompileError(RuntimeError):
    """A kernel TileLang could not lower for this device."""


def arch(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def _pass_configs(target: str) -> dict:
    if target == "sm_90":
        return {}
    return {"tl.disable_warp_specialized": True}


def compiled(builder: Callable, device: torch.device, *args):
    """builder(*args) compiled for `device`'s arch once per process; failures are kept."""
    target = arch(device)
    key = (builder.__module__, builder.__qualname__, target, *args)
    kernel = _CACHE.get(key)
    if kernel is None:
        with _LOCK:
            kernel = _CACHE.get(key)
            if kernel is None:
                try:
                    kernel = tilelang.compile(
                        builder(*args),
                        target={"kind": "cuda", "arch": target},
                        pass_configs=_pass_configs(target),
                    )
                except Exception as err:  # noqa: BLE001
                    kernel = CompileError(f"{builder.__qualname__}{args}: {err}"[:2000])
                _CACHE[key] = kernel
    if isinstance(kernel, CompileError):
        raise kernel
    return kernel


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def chunk_for(n: int, rows: int, device: torch.device, per_sm: int = 2) -> int:
    """Rows per segment tile: about `per_sm` tiles per SM, a multiple of `rows`."""
    target = per_sm * max(1, probe(device).sms)
    return max(rows, cdiv(cdiv(n, target), rows) * rows)


class TilePlan:
    """Segment tiles of N rows: per-tile sample / start / stop and per-sample tile range."""

    def __init__(self, seg: Tensor | None, n: int, b: int, chunk: int, device):
        self.n, self.chunk = n, chunk
        if seg is None:
            b = 1
            cu = torch.arange(2, device=device, dtype=torch.int64) * n
        else:
            ids = torch.arange(b + 1, device=device, dtype=seg.dtype)
            cu = torch.searchsorted(seg, ids).to(torch.int64)
        self.b = b
        nt = (cu[1:] - cu[:-1] + chunk - 1) // chunk
        cut = nt.cumsum(0)
        self.tiles = cdiv(n, chunk) + b
        t = torch.arange(self.tiles, device=device, dtype=torch.int64)
        sample = torch.searchsorted(cut, t, right=True).clamp_max(b - 1)
        start = cu[sample] + (t - (cut[sample] - nt[sample])) * chunk
        stop = torch.minimum(start + chunk, cu[sample + 1])
        stop = torch.where(t < cut[-1], stop, start)
        self.tseg = sample.to(torch.int32)
        self.tstart = start.to(torch.int32)
        self.tstop = stop.to(torch.int32)
        self.tbeg = (cut - nt).to(torch.int32)
        self.tend = cut.to(torch.int32)


_PLANS: dict[tuple, tuple] = {}


_DEAD: set[tuple] = set()


def tuned(
    kernel: str, device, key: str, candidates: list[dict], run: Callable[[dict], object]
):
    """Config for (kernel, key) from the shared tuner; every candidate timed once.
    Configs that do not compile on this arch are dropped from the candidates."""
    live = [c for c in candidates if (kernel, str(sorted(c.items()))) not in _DEAD]
    return select(
        kernel,
        device,
        key,
        lambda: live,
        lambda c: lambda: run(c),
        shortlist=len(live),
        errors=(RuntimeError, CompileError),
    )


def tuned_run(
    kernel: str, device, key: str, candidates: list[dict], run: Callable[[dict], object]
):
    """run(config) for the tuned config; a config that fails to compile is marked dead
    and the next candidate (in order) runs instead."""
    cfg = tuned(kernel, device, key, candidates, run)
    order = [cfg] + [c for c in candidates if c != cfg]
    for c in order:
        if (kernel, str(sorted(c.items()))) in _DEAD:
            continue
        try:
            return run(c)
        except CompileError:
            _DEAD.add((kernel, str(sorted(c.items()))))
    raise CompileError(f"{kernel}: no candidate compiles for {key}")


def tile_plan(
    seg: Tensor | None, n: int, b: int, rows: int, device, per_sm: int = 2
) -> TilePlan:
    """The TilePlan of `seg`, reused while the same tensor is alive (eager only)."""
    chunk = chunk_for(n, rows, device, per_sm)
    if seg is None or torch.compiler.is_compiling():
        return TilePlan(seg, n, b, chunk, device)
    key = (id(seg), chunk)
    hit = _PLANS.get(key)
    if hit is not None:
        ref, version, meta, plan = hit
        if ref() is seg and version == seg._version and meta == (n, b, chunk):
            return plan
    plan = TilePlan(seg, n, b, chunk, device)
    if len(_PLANS) > 64:
        _PLANS.clear()
    _PLANS[key] = (weakref.ref(seg), seg._version, (n, b, chunk), plan)
    return plan


def segsum_prim(d, rows, slots, bd, threads):
    """out[s, b] = sum over the tiles of sample b and their `rows` of part[t, r, s]."""
    b = T.dynamic("b")
    tt = T.dynamic("t")

    @T.prim_func
    def main(
        part: T.Tensor((tt, rows, slots, d), "float32"),
        tbeg: T.Tensor((b,), "int32"),
        tend: T.Tensor((b,), "int32"),
        out: T.Tensor((slots, b, d), "float32"),
    ):
        with T.Kernel(b, T.ceildiv(d, bd), slots, threads=threads) as (sb, cb, s):
            acc = T.alloc_fragment((bd,), "float32")
            T.clear(acc)
            for t in T.serial(tend[sb] - tbeg[sb]):
                for r in T.serial(rows):
                    for j in T.Parallel(bd):
                        acc[j] += part[tbeg[sb] + t, r, s, T.min(cb * bd + j, d - 1)]
            for j in T.Parallel(bd):
                if cb * bd + j < d:
                    out[s, sb, cb * bd + j] = acc[j]

    return main


def segment_sums(part: Tensor, plan: TilePlan, slots: int, d: int) -> Tensor:
    """[slots, B, D] fp32 per-sample sums of part [tiles, rows, slots, D]."""
    out = torch.empty(slots, plan.b, d, device=part.device, dtype=torch.float32)
    kernel = compiled(segsum_prim, part.device, d, part.shape[1], slots, 128, 128)
    kernel(part, plan.tbeg, plan.tend, out)
    return out


def f32(t: Tensor | None, shape, device) -> Tensor:
    """`t` as contiguous fp32, or zeros of `shape` when absent."""
    if t is None:
        return torch.zeros(shape, device=device, dtype=torch.float32)
    return t.to(torch.float32).contiguous()
