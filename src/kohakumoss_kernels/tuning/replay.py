"""Prewarm by replay: the `Bound` calls a model made in a recorded pass (`recording()`)
re-run forward + backward at every planned token count (`Target`), so every tuned op
the run calls is tuned before step 1. See docs/internals/tuning.md#prewarm.
"""

import contextlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from kohakumoss_common.log import get_logger
from kohakumoss_kernels.backends.compile.choice import retune_pending
from kohakumoss_kernels.errors import describe, is_sticky_error
from kohakumoss_kernels.hardware.current import rng_kept
from kohakumoss_kernels.hardware.memory import close_window, open_window, release
from kohakumoss_kernels.recording import Call, Snap
from kohakumoss_kernels.registry import exercise
from kohakumoss_kernels.tuning import tuner
from kohakumoss_kernels.tuning.manifest import Manifest, problem_key

log = get_logger(__name__)


@dataclass
class OpReport:
    """What the replay did for one op: problems run, problems already decided in the
    tables (`cached`, not run), seconds, shortlists timed, bound implementations,
    problems skipped (`oom` of them out of memory), and the device peak (allocated,
    reserved bytes) of its problems."""

    problems: int = 0
    secs: float = 0.0
    timed: int = 0
    bound: set[str] = field(default_factory=set)
    failed: int = 0
    cached: int = 0
    oom: int = 0
    peak_alloc: int = 0
    peak_reserved: int = 0


def _signature(a) -> tuple:
    """Layout of one recorded argument: tensor shape / dtype / grad flag, scalars."""
    match a:
        case Snap():
            return ("t", a.shape, a.dtype, a.requires_grad)
        case tuple() | list():
            return ("s", tuple(_signature(x) for x in a))
        case None | bool() | int() | float() | str() | torch.dtype():
            return ("v", a)
    return ("o", type(a).__qualname__)


def _layout(a, dims: dict[int, int]) -> tuple:
    """What decides an argument's problem key at `dims`: layouts with the token dim
    mapped, kept integer contents, scalars."""
    match a:
        case Snap():
            lead = (dims.get(a.shape[0], a.shape[0]),) if a.shape else ()
            ints = None
            if a.values is not None and not a.floating:
                ints = a.values.cpu().numpy().tobytes()
            return ("t", lead + a.shape[1:], str(a.dtype), a.requires_grad, ints)
        case tuple() | list():
            return ("s", tuple(_layout(x, dims) for x in a))
        case None | bool() | int() | float() | str() | torch.dtype():
            return ("v", repr(a))
    return ("o", type(a).__qualname__)


@dataclass
class Target:
    """One replay point: recorded token counts -> planned ones (`rows`), and for a
    recorded boundary tensor ending at a recorded count the segment lengths that
    replace it (`segments`; absent: the recorded bounds scaled)."""

    rows: dict[int, int]
    segments: dict[int, tuple[int, ...]] = field(default_factory=dict)


def as_target(t: "Target | dict[int, int]") -> Target:
    return t if isinstance(t, Target) else Target(dict(t))


def problem(call: Call, target: Target | dict[int, int], backward: bool) -> str:
    """Manifest key of `call` replayed at `target`."""
    target = as_target(target)
    dims = target.rows
    parts = (
        call.op.backend,
        repr(call.autocast),
        backward,
        tuple(sorted(dims.items())),
        tuple(sorted(target.segments.items())),
        tuple(_layout(a, dims) for a in call.args),
        tuple(sorted((k, _layout(v, dims)) for k, v in call.kwargs.items())),
    )
    return problem_key(call.op.op, parts)


def _device(call: Call) -> torch.device:
    snaps = [a for a in (*call.args, *call.kwargs.values()) if isinstance(a, Snap)]
    return snaps[0].device if snaps else torch.device("cpu")


def unique(calls: Sequence[Call]) -> list[Call]:
    """One call per (op name, backend, argument shapes / dtypes / grad flags /
    scalars): the per-layer `Bound` objects of one op are one problem."""
    seen, out = set(), []
    for c in calls:
        kw = tuple(sorted((k, _signature(v)) for k, v in c.kwargs.items()))
        key = (c.op.op, c.op.backend, tuple(_signature(a) for a in c.args), kw)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _bounds_like(s: Snap, dims: dict[int, int]) -> bool:
    """A kept integer [B+1] tensor of nondecreasing bounds from 0 to a token count."""
    t = s.values
    return (
        t is not None
        and not s.floating
        and t.dim() == 1
        and t.numel() >= 2
        and int(t[0]) == 0
        and int(t[-1]) in dims
        and bool((t[1:] >= t[:-1]).all())
    )


def _bounds(s: Snap, target: Target) -> Tensor:
    """Bounds `s` at `target`: the target's segment lengths for this total, else the
    recorded bounds scaled to the new total (rounded to nearest, integer math)."""
    t = s.values
    n0 = int(t[-1])
    lengths = target.segments.get(n0)
    if lengths is not None:
        new = torch.tensor((0, *lengths), dtype=torch.int64).cumsum(0)
        return new.to(device=t.device, dtype=t.dtype)
    n = target.rows[n0]
    new = ((t.long() * (2 * n) + n0) // (2 * n0)).to(t.dtype)
    new[-1] = n
    return new


def _tensor(a, dims: dict[int, int]):
    """A fresh argument from a recorded one: snaps materialized on their recorded
    device, leading token dims resampled; other values unchanged."""
    match a:
        case Snap():
            rows = dims.get(a.shape[0]) if a.shape else None
            t = a.tensor(rows)
            return t.requires_grad_(a.requires_grad and a.floating)
        case tuple() | list():
            return type(a)(_tensor(x, dims) for x in a)
    return a


def scale(call: Call, target: Target | dict[int, int]) -> tuple[tuple, dict]:
    """`call`'s arguments materialized from their snapshots at `target`: every
    recorded token count `n` replaced by `rows[n]`, boundary tensors and the longest
    segment (the int argument right after a boundary tensor, the ops' `max_seqlen`)
    with them."""
    target = as_target(target)
    dims = target.rows
    args = []
    longest: dict[int, tuple[int, int]] = {}
    for i, a in enumerate(call.args):
        if isinstance(a, Snap) and _bounds_like(a, dims):
            new = _bounds(a, target)
            old = a.values
            spans = (int((old[1:] - old[:-1]).max()), int((new[1:] - new[:-1]).max()))
            longest[i + 1] = spans
            args.append(new)
            continue
        args.append(_tensor(a, dims))
    for i, (was, now) in longest.items():
        if i < len(args) and type(args[i]) is int and args[i] == was:
            args[i] = now
    return tuple(args), {k: _tensor(v, dims) for k, v in call.kwargs.items()}


def _run(fn: Callable, args: tuple, kwargs: dict, backward: bool) -> None:
    out = fn(*args, **kwargs)
    outs = out if isinstance(out, tuple) else (out,)
    if backward:
        sums = [o.float().sum() for o in outs if o.requires_grad]
        if sums:
            torch.autograd.backward(sums)


def _problem(
    call: Call, target: Target, backward: bool, manifest: Manifest, rep: OpReport
) -> None:
    """Replay `call` at `target` unless decided, through the op's `Bound` as build time
    (`registry.exercise`): a backend that cannot build or launch, forward or backward,
    falls the op (and every live op of the same query) back before step 1. A failure
    left after that (out-of-memory included, after a cache release) is skipped and
    counted."""
    key = problem(call, target, backward)
    if manifest.covered(key):
        rep.cached += 1
        return
    args, kwargs = scale(call, target)
    grad = backward and any(isinstance(a, Tensor) and a.requires_grad for a in args)
    began, before = time.perf_counter(), timed_kernels()
    cast = (
        torch.autocast(call.autocast[0], dtype=call.autocast[1])
        if call.autocast is not None
        else contextlib.nullcontext()
    )
    short = False
    try:
        with manifest.running(key), cast:
            with torch.enable_grad() if grad else torch.no_grad():
                exercise(call.op, lambda op: _run(op, args, kwargs, grad))
                decide = getattr(call.op, "decide", None)
                if decide is not None:
                    rep.bound.add(decide(*args, **kwargs))
    except torch.OutOfMemoryError:
        short = True
    except Exception as err:  # noqa: BLE001
        if is_sticky_error(err):
            raise
        rep.failed += 1
        log.warning(f"prewarm replay of {call.op.op} skipped: {describe(err)}")
        return
    if short:
        del args, kwargs
        release(_device(call))
        rep.failed += 1
        rep.oom += 1
        log.warning(f"prewarm replay of {call.op.op} skipped: out of memory")
        return
    rep.problems += 1
    rep.secs += time.perf_counter() - began
    rep.timed += timed_kernels() - before


def replay(
    calls: Sequence[Call],
    targets: Sequence[Target | dict[int, int]],
    backward: bool = True,
) -> dict[str, OpReport]:
    """Run each unique recorded call at every point of `targets` (forward +
    backward where an argument needs grad and `backward`) through the op's `Bound`,
    one op at a time (its device peak measured, free blocks returned to the driver
    after it), then decide the queued kernel-vs-compile problems. A backend failure
    rebinds the op (`_problem`); a problem still failing is skipped and counted. The
    synthetic inputs never move the caller's generators (`rng_kept`). Returns a report
    per op name."""
    reports: dict[str, OpReport] = {}
    ops = {}
    manifests: dict[str, Manifest] = {}
    groups: dict[str, list[Call]] = {}
    for call in unique(calls):
        groups.setdefault(call.op.op, []).append(call)
    for name, group in groups.items():
        rep = reports.setdefault(name, OpReport())
        ops[name] = group[0].op
        devices = {str(_device(c)): _device(c) for c in group}
        windows = []
        for device in devices.values():
            release(device)
            windows.append(open_window(device))
        for call in group:
            device = _device(call)
            manifest = manifests.setdefault(str(device), Manifest(device))
            for target in targets:
                with rng_kept([device]):
                    _problem(call, target, backward, manifest, rep)
        for w in windows:
            alloc, held = close_window(w)
            rep.peak_alloc = max(rep.peak_alloc, alloc)
            rep.peak_reserved = max(rep.peak_reserved, held)
            release(w.device)
    retune_pending()
    for manifest in manifests.values():
        manifest.commit()
    for name, op in ops.items():
        if not reports[name].bound:
            reports[name].bound.add(op.backend)
    return reports


def timed_kernels() -> int:
    """How many shortlists this process has timed so far (all kernels)."""
    return sum(tuner.TIMED.values())
