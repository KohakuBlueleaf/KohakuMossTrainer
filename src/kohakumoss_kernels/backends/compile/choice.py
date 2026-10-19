"""Kernel-vs-compile choice per op and problem shape, bound once, never timed in a call.

The choices of a registered op are its own backend's kernel (`kernel`) and `compile`.
A call looks its problem up: a bound callable, else the settled per-card result of an
earlier measurement, else the kernel (queued for the retune until `window.seal`).
`retune_pending()` (end of the prewarm replay and the retune steps, outside any
autograd graph) measures the queued problems and stores the winners; the next call of
each binds its winner; `decide(*args)` measures one problem directly. See
docs/internals/tuning.md#kernel-vs-compile.
"""

import time
import weakref
from collections.abc import Callable

import torch
from torch import Tensor

from kohakumoss_common.log import get_logger, hl
from kohakumoss_kernels.backends.compile.ops import CompiledOp, compiled_op
from kohakumoss_kernels.dtypes import short
from kohakumoss_kernels.errors import BUILD_ERRORS, BackendUnavailable, unavailable
from kohakumoss_kernels.hardware.current import rng_kept
from kohakumoss_kernels.hardware.memory import release
from kohakumoss_kernels.tuning import decisions
from kohakumoss_kernels.tuning.buckets import TOKEN_FLOOR, pow2
from kohakumoss_kernels.tuning.tuner import (
    decided,
    get_mode,
    note,
    on_reset,
    select,
    settled,
    timing_allowed,
)
from kohakumoss_kernels.tuning.window import sealed, timing_here

log = get_logger(__name__)
KERNEL = {"impl": "kernel"}
CHOICES = ("kernel", "compile")
_OPS: "weakref.WeakSet[KernelOrCompile]" = weakref.WeakSet()


def _device(args) -> torch.device:
    return next(a.device for a in args if isinstance(a, Tensor))


def _no_grad_run(fn: Callable, args) -> None:
    with torch.no_grad():
        fn(*[a.detach() if isinstance(a, Tensor) else a for a in args])


def _bucketed(shape: torch.Size) -> tuple:
    """`shape` with its leading (token) dim in its problem-key bucket (`shape_key`)."""
    return (pow2(shape[0], TOKEN_FLOOR), *shape[1:]) if shape else ()


def _call_key(args) -> tuple:
    """Per-call binding key: the first tensor's device, grad mode, intra-op threads and
    autocast dtype, then per argument tensor shape (token dim bucketed as the problem
    key buckets it) / dtype / grad flag or the value (at least what a compiled
    callable's call pattern holds)."""
    first = next((a for a in args if isinstance(a, Tensor)), None)
    kind = "cpu" if first is None else first.device.type
    cast = torch.get_autocast_dtype(kind) if torch.is_autocast_enabled(kind) else None
    state = (
        None if first is None else first.device,
        torch.is_grad_enabled(),
        torch.get_num_threads(),
        cast,
    )
    return state + tuple(
        (_bucketed(a.shape), a.dtype, a.requires_grad) if isinstance(a, Tensor) else a
        for a in args
    )


def _layout(a):
    """What a queued problem keeps of one argument: integer tensors cloned, floating
    tensors as (shape, dtype, device, requires_grad), everything else as is."""
    if isinstance(a, Tensor):
        if a.is_floating_point():
            return ("float", tuple(a.shape), a.dtype, a.device, a.requires_grad)
        return ("int", a.detach().clone())
    return ("value", a)


def _rebuild(layout):
    match layout:
        case ("float", shape, dtype, device, grad):
            return torch.randn(shape, device=device, dtype=dtype).requires_grad_(grad)
        case ("int", t):
            return t
        case ("value", v):
            return v


def _step(fn: Callable, args) -> Callable[[], None]:
    """Forward, plus backward over the arguments that require grad (the work the
    training call does); forward only when none does."""
    leaves = [
        (
            a.detach().requires_grad_(a.requires_grad and a.is_floating_point())
            if isinstance(a, Tensor)
            else a
        )
        for a in args
    ]
    if not any(isinstance(a, Tensor) and a.requires_grad for a in leaves):
        return lambda: fn(*leaves)

    def run():
        out = fn(*leaves)
        outs = out if isinstance(out, tuple) else (out,)
        torch.autograd.backward(outs, [torch.ones_like(o) for o in outs])
        for leaf in leaves:
            if isinstance(leaf, Tensor):
                leaf.grad = None

    return run


class KernelOrCompile:
    """An op bound to its backend kernel or its compile path per problem shape."""

    def __init__(
        self, backend: str, op_name: str, kernel_fn: Callable, key_fn: Callable
    ):
        self.name = f"choice_{backend}_{op_name}"
        self.backend = backend
        self.op_name = op_name
        self.key_fn = key_fn
        self.impls = {"kernel": kernel_fn, "compile": compiled_op(op_name)}
        self.probed: dict[str, BackendUnavailable | None] = {}
        self.bound: dict[tuple, Callable] = {}
        self.pending: dict[tuple, list] = {}
        self.pinned: str | None = None
        self.choice: str | None = None
        self.__wrapped__ = kernel_fn
        self.__name__ = getattr(kernel_fn, "__name__", op_name)
        _OPS.add(self)

    def problem(self, args) -> str:
        """Stored-table key: bucketed shapes plus which tensors require grad."""
        flags = "".join(
            "1" if isinstance(a, Tensor) and a.requires_grad else "0" for a in args
        )
        return f"{self.key_fn(*args)}-g{flags}"

    def _probe(self, args) -> None:
        """Run the kernel once per device without autograd; a build failure raises
        `BackendUnavailable` (the registry falls back), anything else propagates."""
        dev = str(_device(args))
        if dev not in self.probed:
            try:
                _no_grad_run(self.impls["kernel"], args)
                self.probed[dev] = None
            except Exception as err:  # noqa: BLE001
                self.probed[dev] = unavailable(self.op_name, err, dev)
        if self.probed[dev] is not None:
            raise self.probed[dev]

    def _impl(self, impl: str, args) -> Callable:
        """The callable of `impl` for this call: a `CompiledOp`'s runner of the call's
        pattern, else the implementation itself."""
        run = self.impls[impl]
        return run.resolved(args) if isinstance(run, CompiledOp) else run

    def _lookup(self, args, qk) -> Callable:
        if self.pinned is not None:
            fn = self._impl(self.pinned, args)
        elif torch.compiler.is_compiling():
            return self.impls["kernel"]
        else:
            self._probe(args)
            queues = get_mode() == "table" and not sealed()
            key = self.problem(args)
            note(self.name, key)
            row = settled(self.name, _device(args), key)
            if row is not None and row.get("impl") not in CHOICES:
                row = None
            impl = row["impl"] if row is not None else "kernel"
            if row is not None:
                decisions.record(self.name, key, "table", {"impl": impl})
            elif queues:
                self.pending[qk] = [_layout(a) for a in args]
                decisions.undecided(self.name, key)
            else:
                decisions.record(self.name, key, "default", KERNEL)
            fn = self._impl(impl, args)
        self.bound[qk] = fn
        return fn

    def __call__(self, *args):
        if torch.compiler.is_compiling():
            return self.impls[self.pinned or "kernel"](*args)
        qk = _call_key(args)
        fn = self.bound.get(qk)
        if fn is None:
            fn = self._lookup(args, qk)
        return fn(*args)

    def decide(self, *args) -> str:
        """Measure this problem (or read its stored result), store and bind the winner.
        Frozen or deferring outside `timing_here`, the kernel binds untimed and unstored
        (`tuner._bind_untimed`). Where the context forbids timing an autograd launch the
        kernel is returned unbound and the problem stays queued."""
        device = _device(args)
        key = self.problem(args)
        cfg = select(
            self.name,
            device,
            key,
            lambda: [{"impl": c} for c in CHOICES],
            lambda c: _step(self.impls[c["impl"]], args),
            shortlist=len(CHOICES),
            errors=BUILD_ERRORS,
            autograd=True,
        )
        self.choice = cfg["impl"]
        if decided(self.name, device, key):
            qk = _call_key(args)
            self.bound[qk] = self._impl(self.choice, args)
            self.pending.pop(qk, None)
        return self.choice

    def bind(self, choice: str) -> None:
        """Pin `choice` for every problem of this op."""
        self.pinned = choice
        self.choice = choice
        self.bound.clear()
        self.pending.clear()

    def retune(self) -> int:
        """Measure every queued problem on rebuilt inputs; a decided problem's next call
        binds the winner. A problem whose backend cannot run is dropped; one whose
        inputs do not fit in memory stays on the kernel (dropped, logged). Returns how
        many were decided."""
        done = 0
        for qk, layouts in list(self.pending.items()):
            match self._retune_one(layouts):
                case "queued":
                    continue
                case "decided":
                    # drop the provisional kernel binding; keep the winner `decide` bound
                    stale = self.bound.get(qk) is self.impls["kernel"]
                    if stale and self.choice != "kernel":
                        self.bound.pop(qk, None)
                    done += 1
                case "oom":
                    release(qk[0])
                    log.warning(
                        f"{self.name}: the inputs of a queued problem do not fit in "
                        "memory; it stays on the kernel"
                    )
            self.pending.pop(qk, None)
        return done

    def _retune_one(self, layouts) -> str:
        """Decide one queued problem: "decided", "queued" (not settled here),
        "unavailable" (its backend cannot run) or "oom"."""
        try:
            args = [_rebuild(s) for s in layouts]
            self.decide(*args)
            return (
                "decided"
                if decided(self.name, _device(args), self.problem(args))
                else "queued"
            )
        except BackendUnavailable:
            return "unavailable"
        except torch.OutOfMemoryError:
            return "oom"


def kernel_or_compile(
    backend: str, op_name: str, kernel_fn: Callable, key_fn: Callable
) -> KernelOrCompile:
    """The registry entry of a block-op kernel: chosen per shape against `compile`."""
    return KernelOrCompile(backend, op_name, kernel_fn, key_fn)


def reset() -> None:
    """Forget every binding and queued problem (tuner reset, broadcast, tests)."""
    for op in list(_OPS):
        op.bound.clear()
        op.pending.clear()
        op.pinned = None
        op.choice = None


on_reset(reset)


def pending_count() -> int:
    """Queued kernel-vs-compile problems of every op (what `retune_pending` decides)."""
    return sum(len(op.pending) for op in list(_OPS))


def drop_pending() -> int:
    """Forget every queued problem (each stays on its provisional kernel binding);
    returns how many there were."""
    count = pending_count()
    for op in list(_OPS):
        op.pending.clear()
    return count


def retune_pending() -> int:
    """Decide every queued problem of every op (end of the prewarm replay, the retune
    steps, after an owned pipeline's generate), returning memory to the driver only
    when a problem's inputs do not fit; nothing is measured where the context forbids
    timing an autograd launch."""
    if not timing_allowed(autograd=True):
        return 0
    devices = {
        s[3]
        for op in list(_OPS)
        for layouts in op.pending.values()
        for s in layouts
        if s[0] == "float"
    }
    start = time.perf_counter()
    with torch.enable_grad(), rng_kept(devices), timing_here():
        measured = sum(op.retune() for op in list(_OPS))
    if measured:
        took = time.perf_counter() - start
        log.debug(
            f"measured {hl(measured)} kernel-vs-compile problems in "
            f"{hl(f'{took:.2f}s')}"
        )
        decisions.flush(f"after a retune ({took:.2f}s)")
    return measured


def shape_key(*args) -> str:
    """Problem key: tensor shapes bucketed on the leading (token) dim, dtypes, scalars."""
    parts = []
    for a in args:
        if isinstance(a, Tensor):
            dims = list(a.shape)
            if dims:
                dims[0] = pow2(dims[0], TOKEN_FLOOR)
            parts.append(f"{'x'.join(map(str, dims))}{short(a.dtype)}")
        elif a is None:
            parts.append("none")
        elif isinstance(a, float):
            parts.append("r1" if a == 1.0 else "rx")
        else:
            parts.append(str(a))
    return "|".join(parts)
