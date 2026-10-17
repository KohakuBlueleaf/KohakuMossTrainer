"""Shared pieces of the TileLang block ops: compile cache, segment tile tables.

Kernels take caller-allocated outputs (TileLang-allocated outputs lose autograd when a
custom op returns them) and a symbolic token count, so one compile serves every N.
Segment tile tables (sample, first row, end row per tile) are built with device torch
ops from `seg` (no host sync) and cached per `seg` tensor for the rest of the step.
"""

import math
import threading
import weakref
from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch
from torch import Tensor

from kohakumoss_kernels.backends.tilelang._df import df_add
from kohakumoss_kernels.hardware.current import on_device
from kohakumoss_kernels.hardware.device import probe
from kohakumoss_kernels.tuning.tuner import select

DTYPE = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}
F32 = "float32"
I32 = "int32"

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


class OnDevice:
    """A compiled kernel launched with its device current."""

    __slots__ = ("kernel", "device")

    def __init__(self, kernel, device: torch.device):
        self.kernel, self.device = kernel, device

    def __call__(self, *args):
        with on_device(self.device):
            return self.kernel(*args)


def compiled(builder: Callable, device: torch.device, *args) -> OnDevice:
    """builder(*args) compiled for `device`'s arch once per process (failures are
    kept), launched on `device`."""
    return OnDevice(_compiled(builder, device, *args), device)


def _compiled(builder: Callable, device: torch.device, *args):
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


def segsum_groups_prim(d, rows, slots, bd, threads):
    """Pass 1 of the pair sums: group g of sample b sums `ch` consecutive (tile, row)
    entries of the fp32 pairs part[t, r, s, :, :] into mid [g, slots, 2, b, d]."""
    b = T.dynamic("b")
    tt = T.dynamic("t")
    gg = T.dynamic("g")

    @T.prim_func
    def main(
        part: T.Tensor((tt, rows, slots, 2, d), F32),
        tbeg: T.Tensor((b,), I32),
        tend: T.Tensor((b,), I32),
        mid: T.Tensor((gg, slots, 2, b, d), F32),
        ch: T.int32,
    ):
        with T.Kernel(b, T.ceildiv(d, bd), gg * slots, threads=threads) as (
            sb,
            cb,
            gs,
        ):
            hi = T.alloc_fragment((bd,), F32)
            lo = T.alloc_fragment((bd,), F32)
            gi = gs // slots
            s = gs % slots
            ubeg = tbeg[sb] * rows + gi * ch
            uend = T.min(tend[sb] * rows, ubeg + ch)
            T.clear(hi)
            T.clear(lo)
            for u in T.serial(T.max(uend - ubeg, 0)):
                for j in T.Parallel(bd):
                    col = T.min(cb * bd + j, d - 1)
                    t = (ubeg + u) // rows
                    r = (ubeg + u) % rows
                    h, lw = df_add(
                        hi[j], lo[j], part[t, r, s, 0, col], part[t, r, s, 1, col]
                    )
                    hi[j] = h
                    lo[j] = lw
            for j in T.Parallel(bd):
                if cb * bd + j < d:
                    mid[gi, s, 0, sb, cb * bd + j] = hi[j]
                    mid[gi, s, 1, sb, cb * bd + j] = lo[j]

    return main


def segsum_final_prim(d, slots, bd, threads, rounded):
    """Pass 2: pair sums over the groups of mid [g, slots, 2, b, d]: out [slots, 2,
    b, d] pairs, or [slots, b, d] rounded to fp32."""
    b = T.dynamic("b")
    gg = T.dynamic("g")
    shape = (slots, b, d) if rounded else (slots, 2, b, d)

    @T.prim_func
    def main(mid: T.Tensor((gg, slots, 2, b, d), F32), out: T.Tensor(shape, F32)):
        with T.Kernel(b, T.ceildiv(d, bd), slots, threads=threads) as (sb, cb, s):
            hi = T.alloc_fragment((bd,), F32)
            lo = T.alloc_fragment((bd,), F32)
            T.clear(hi)
            T.clear(lo)
            for g in T.serial(gg):
                for j in T.Parallel(bd):
                    col = T.min(cb * bd + j, d - 1)
                    h, lw = df_add(
                        hi[j], lo[j], mid[g, s, 0, sb, col], mid[g, s, 1, sb, col]
                    )
                    hi[j] = h
                    lo[j] = lw
            for j in T.Parallel(bd):
                if cb * bd + j < d:
                    if rounded:
                        out[s, sb, cb * bd + j] = T.ieee_add(hi[j], lo[j])
                    else:
                        out[s, 0, sb, cb * bd + j] = hi[j]
                        out[s, 1, sb, cb * bd + j] = lo[j]

    return main


def segment_sums(part: Tensor, plan: TilePlan, rounded: bool) -> Tensor:
    """Per-sample pair sums of part [tiles, rows, slots, 2, D] (fp32 pairs): pairs
    [slots, 2, B, D], or [slots, B, D] rounded to fp32 when `rounded`."""
    return pair_sums(part, plan.tbeg, plan.tend, plan.b, rounded)


def pair_sums(part: Tensor, tbeg: Tensor, tend: Tensor, b: int, rounded: bool):
    """`segment_sums` over explicit per-sample tile ranges [tbeg, tend): groups of
    about sqrt(entries) (tile, row) entries summed in parallel, then the groups."""
    tiles, rows, slots, _, d = part.shape
    entries = max(1, tiles * rows)
    ch = max(8, math.isqrt(entries))
    groups = cdiv(entries, ch)
    bd = min(128, 1 << max(0, (d - 1).bit_length()))
    dev = part.device
    mid = torch.empty(groups, slots, 2, b, d, device=dev, dtype=torch.float32)
    compiled(segsum_groups_prim, dev, d, rows, slots, bd, 128)(
        part, tbeg, tend, mid, ch
    )
    shape = (slots, b, d) if rounded else (slots, 2, b, d)
    out = torch.empty(shape, device=dev, dtype=torch.float32)
    compiled(segsum_final_prim, dev, d, slots, bd, 128, rounded)(mid, out)
    return out


def total(part: Tensor) -> Tensor:
    """fp32 [slots, D] sums over every tile and row of part [T, rows, slots, 2, D]."""
    dev = part.device
    tbeg = torch.zeros(1, device=dev, dtype=torch.int32)
    tend = torch.full((1,), part.shape[0], device=dev, dtype=torch.int32)
    return pair_sums(part, tbeg, tend, 1, True)[:, 0]


def f32(t: Tensor | None, shape, device) -> Tensor:
    """`t` as contiguous fp32, or zeros of `shape` when absent."""
    if t is None:
        return torch.zeros(shape, device=device, dtype=torch.float32)
    return t.to(torch.float32).contiguous()


_ZEROS: dict[tuple, Tensor] = {}


def per_sample(t: Tensor | None, b: int, d: int, device) -> tuple[Tensor, int, int]:
    """A per-sample [B, D] tensor as (flat fp32 view over its storage, row stride,
    column stride): any strides, no copy when it is fp32; shared zeros when absent."""
    if t is None:
        key = (b, d, str(device))
        if key not in _ZEROS:
            _ZEROS[key] = torch.zeros(b * d, device=device, dtype=torch.float32)
        return _ZEROS[key], d, 1
    if t.dtype != torch.float32:
        t = t.float()
    rs, cs = t.stride()
    length = (t.shape[0] - 1) * rs + (t.shape[1] - 1) * cs + 1
    return t.as_strided((length,), (1,), t.storage_offset()), rs, cs
