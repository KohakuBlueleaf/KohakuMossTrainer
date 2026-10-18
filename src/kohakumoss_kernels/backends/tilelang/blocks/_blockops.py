"""Shared pieces of the TileLang block ops: compile cache, tuned launches, segment tile
tables, per-sample operand views.

Kernels take caller-allocated outputs and a symbolic token count, so one compile serves
every N. Segment tile tables (sample, first row, end row per tile) are built with device
torch ops from `seg` (no host sync) and cached per `seg` tensor. See
docs/internals/kernel-dev.md.
"""

import math
import threading
from collections.abc import Callable

import tilelang
import tilelang.language as T
import torch
from torch import Tensor

from kohakumoss_kernels.backends._memo import TensorMemo
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.hardware.current import on_device
from kohakumoss_kernels.hardware.device import measure_rates
from kohakumoss_kernels.tuning.buckets import cdiv, chunk_rows
from kohakumoss_kernels.tuning.tuner import select

DTYPE = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}
F32 = "float32"
I32 = "int32"
FP16_SUPPORTS = requires(kinds=("cuda",), min_sm=70, dtypes=(torch.float16,))

_CACHE: dict[tuple, object] = {}
_LOCK = threading.Lock()
_PLANS = TensorMemo()
_DEAD: set[tuple] = set()
_ZEROS: dict[tuple, Tensor] = {}


class CompileError(RuntimeError):
    """A kernel TileLang could not lower for this device."""


def arch(device: torch.device) -> str:
    """TileLang target arch of a CUDA device (`sm_XY`)."""
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def compile_once(
    key: tuple, build: Callable[[], object], error: type[Exception] | None = None
):
    """`build()` once per process under `key` (thread-safe). With `error`, a failure is
    kept and raised as `error` on every lookup; without, it propagates and is retried.
    """
    kernel = _CACHE.get(key)
    if kernel is None:
        with _LOCK:
            kernel = _CACHE.get(key)
            if kernel is None:
                try:
                    kernel = build()
                except Exception as err:  # noqa: BLE001
                    if error is None:
                        raise
                    kernel = error(f"{key}: {err}"[:2000])
                _CACHE[key] = kernel
    if isinstance(kernel, Exception):
        raise type(kernel)(*kernel.args)
    return kernel


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
    """builder(*args) compiled for `device`'s arch once per process (a failure is kept
    and raised as `CompileError`), launched on `device`."""
    target = arch(device)
    kernel = compile_once(
        (builder.__module__, builder.__qualname__, target, *args),
        lambda: tilelang.compile(
            builder(*args),
            target={"kind": "cuda", "arch": target},
            pass_configs=_pass_configs(target),
        ),
        CompileError,
    )
    return OnDevice(kernel, device)


def _config_id(kernel: str, config: dict) -> tuple:
    return kernel, str(sorted(config.items()))


def mark_dead(kernel: str, config: dict) -> None:
    """Drop `config` from `kernel`'s candidates (it failed to compile on this arch)."""
    _DEAD.add(_config_id(kernel, config))


def _live(kernel: str, key: str, candidates: list[dict]) -> list[dict]:
    """The candidates of `kernel` not marked as failing to compile."""
    live = [c for c in candidates if _config_id(kernel, c) not in _DEAD]
    if not live:
        raise CompileError(f"{kernel}: no candidate compiles for {key}")
    return live


def tuned(
    kernel: str, device, key: str, candidates: list[dict], run: Callable[[dict], object]
):
    """Config for (kernel, key) from the shared tuner; every candidate timed once (also
    inside a backward: the launches record no autograd). Configs that do not compile
    on this arch are dropped from the candidates."""
    return select(
        kernel,
        device,
        key,
        lambda: _live(kernel, key, candidates),
        lambda c: lambda: run(c),
        shortlist=len(candidates),
        errors=(RuntimeError,),
        in_backward=True,
    )


def tuned_run(
    kernel: str, device, key: str, candidates: list[dict], run: Callable[[dict], object]
):
    """run(config) for the tuned config; a config that fails to compile is marked dead
    and the next candidate (in order) runs instead."""
    cfg = tuned(kernel, device, key, candidates, run)
    for c in (cfg, *candidates):
        ident = _config_id(kernel, c)
        if ident in _DEAD:
            continue
        try:
            return run(c)
        except CompileError:
            _DEAD.add(ident)
    raise CompileError(f"{kernel}: no candidate compiles for {key}")


def chunk_for(n: int, rows: int, device: torch.device, per_sm: int = 2) -> int:
    """Rows per segment tile: about `per_sm` tiles per SM, a multiple of `rows`."""
    return chunk_rows(n, rows, max(1, measure_rates(device).sms), per_sm)


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


def tile_plan(
    seg: Tensor | None, n: int, b: int, rows: int, device, per_sm: int = 2
) -> TilePlan:
    """The TilePlan of `seg`, reused while the same tensor is alive (eager only)."""
    chunk = chunk_for(n, rows, device, per_sm)
    if seg is None or torch.compiler.is_compiling():
        return TilePlan(seg, n, b, chunk, device)
    return _PLANS.get(seg, (n, b, chunk), lambda: TilePlan(seg, n, b, chunk, device))


def segsum_groups_prim(d, rows, slots, bd, threads):
    """Pass 1 of the segment sums: group g of sample b sums `ch` consecutive (tile, row)
    entries of the fp32 partials part[t, r, s, :] into mid [g, slots, b, d]."""
    b = T.dynamic("b")
    tt = T.dynamic("t")
    gg = T.dynamic("g")

    @T.prim_func
    def main(
        part: T.Tensor((tt, rows, slots, d), F32),
        tbeg: T.Tensor((b,), I32),
        tend: T.Tensor((b,), I32),
        mid: T.Tensor((gg, slots, b, d), F32),
        ch: T.int32,
    ):
        with T.Kernel(b, T.ceildiv(d, bd), gg * slots, threads=threads) as (
            sb,
            cb,
            gs,
        ):
            acc = T.alloc_fragment((bd,), F32)
            gi = gs // slots
            s = gs % slots
            ubeg = tbeg[sb] * rows + gi * ch
            uend = T.min(tend[sb] * rows, ubeg + ch)
            T.clear(acc)
            for u in T.serial(T.max(uend - ubeg, 0)):
                for j in T.Parallel(bd):
                    col = T.min(cb * bd + j, d - 1)
                    t = (ubeg + u) // rows
                    r = (ubeg + u) % rows
                    acc[j] = acc[j] + part[t, r, s, col]
            for j in T.Parallel(bd):
                if cb * bd + j < d:
                    mid[gi, s, sb, cb * bd + j] = acc[j]

    return main


def segsum_final_prim(d, slots, bd, threads):
    """Pass 2: fp32 sums over the groups of mid [g, slots, b, d] into out [slots, b, d]."""
    b = T.dynamic("b")
    gg = T.dynamic("g")

    @T.prim_func
    def main(mid: T.Tensor((gg, slots, b, d), F32), out: T.Tensor((slots, b, d), F32)):
        with T.Kernel(b, T.ceildiv(d, bd), slots, threads=threads) as (sb, cb, s):
            acc = T.alloc_fragment((bd,), F32)
            T.clear(acc)
            for g in T.serial(gg):
                for j in T.Parallel(bd):
                    col = T.min(cb * bd + j, d - 1)
                    acc[j] = acc[j] + mid[g, s, sb, col]
            for j in T.Parallel(bd):
                if cb * bd + j < d:
                    out[s, sb, cb * bd + j] = acc[j]

    return main


def segment_sums(part: Tensor, plan: TilePlan) -> Tensor:
    """Per-sample fp32 sums [slots, B, D] of part [tiles, rows, slots, D]."""
    return range_sums(part, plan.tbeg, plan.tend, plan.b)


def range_sums(part: Tensor, tbeg: Tensor, tend: Tensor, b: int) -> Tensor:
    """`segment_sums` over explicit per-sample tile ranges [tbeg, tend): groups of
    about sqrt(entries) (tile, row) entries summed in parallel, then the groups."""
    tiles, rows, slots, d = part.shape
    entries = max(1, tiles * rows)
    ch = max(8, math.isqrt(entries))
    groups = cdiv(entries, ch)
    bd = min(128, 1 << max(0, (d - 1).bit_length()))
    dev = part.device
    mid = torch.empty(groups, slots, b, d, device=dev, dtype=torch.float32)
    compiled(segsum_groups_prim, dev, d, rows, slots, bd, 128)(
        part, tbeg, tend, mid, ch
    )
    out = torch.empty(slots, b, d, device=dev, dtype=torch.float32)
    compiled(segsum_final_prim, dev, d, slots, bd, 128)(mid, out)
    return out


def total(part: Tensor) -> Tensor:
    """fp32 [slots, D] sums over every tile and row of part [T, rows, slots, D]."""
    dev = part.device
    tbeg = torch.zeros(1, device=dev, dtype=torch.int32)
    tend = torch.full((1,), part.shape[0], device=dev, dtype=torch.int32)
    return range_sums(part, tbeg, tend, 1)[:, 0]


def f32(t: Tensor | None, shape, device) -> Tensor:
    """`t` as contiguous fp32, or zeros of `shape` when absent."""
    if t is None:
        return torch.zeros(shape, device=device, dtype=torch.float32)
    return t.to(torch.float32).contiguous()


def segment_ids(seg: Tensor | None, n: int, device) -> Tensor:
    """`seg` as contiguous int32, or zeros [n] (one sample) when absent."""
    if seg is None:
        return torch.zeros(n, device=device, dtype=torch.int32)
    return seg.to(torch.int32).contiguous()


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


def head_dims(names: tuple[str, ...], allowed: tuple[int, ...]):
    """A `requires(extra=...)` check: every static `names` entry given is in `allowed`."""

    def check(query) -> str | None:
        for name in names:
            d = query.get(name)
            if d is not None and d not in allowed:
                return f"{name} {d} not in {allowed}"
        return None

    return check
