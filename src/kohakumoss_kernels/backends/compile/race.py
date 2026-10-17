"""Kernel-vs-compile choice per op and problem shape, bound once, never timed in a call.

A call looks its problem up: a bound callable, else the stored per-card result of an
earlier measurement, else the kernel (and the problem is queued). Nothing is timed in
the call path, so a forward and its checkpoint recompute always bind the same
implementation. `retune_pending()` (called between steps, outside any autograd
graph) measures queued problems on synthetic inputs, stores the winners and rebinds;
`decide(*args)` measures one problem directly (benches, prewarm). TUNE=off disables
both.
"""

import weakref
from collections.abc import Callable

import torch
from torch import Tensor

from kohakumoss_kernels.backends.compile.ops import compiled_op
from kohakumoss_kernels.errors import BUILD_ERRORS, BackendUnavailable, unavailable
from kohakumoss_kernels.tuning.tuner import (
    decided,
    get_mode,
    on_reset,
    select,
    table,
    timing_allowed,
)

KERNEL = {"impl": "kernel"}
COMPILE = {"impl": "compile"}
_OPS: "weakref.WeakSet[RacedOp]" = weakref.WeakSet()


def _device(args) -> torch.device:
    return next(a.device for a in args if isinstance(a, Tensor))


def _quick_key(args) -> tuple:
    """Cheap per-call key: shapes, dtypes and grad flags of tensors, scalars as given."""
    return tuple(
        (
            (tuple(a.shape), a.dtype, a.requires_grad)
            if isinstance(a, Tensor)
            else (a is None, a == 1.0 if isinstance(a, float) else a)
        )
        for a in args
    )


def _spec(a):
    """What a queued problem keeps of one argument: integer tensors (layout) cloned,
    floating tensors as (shape, dtype, device, requires_grad), everything else as is."""
    if isinstance(a, Tensor):
        if a.is_floating_point():
            return ("float", tuple(a.shape), a.dtype, a.device, a.requires_grad)
        return ("int", a.detach().clone())
    return ("value", a)


def _rebuild(spec):
    match spec:
        case ("float", shape, dtype, device, grad):
            return torch.randn(shape, device=device).to(dtype).requires_grad_(grad)
        case ("int", t):
            return t
        case ("value", v):
            return v


def _step(fn: Callable, args) -> Callable[[], None]:
    """Forward, plus backward over the arguments that require grad (the same work the
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


class RacedOp:
    """An op bound to its kernel or its compile path per problem shape."""

    def __init__(
        self, backend: str, op_name: str, kernel_fn: Callable, key_fn: Callable
    ):
        self.name = f"race_{backend}_{op_name}"
        self.op_name = op_name
        self.key_fn = key_fn
        self.impls = {"kernel": kernel_fn, "compile": compiled_op(op_name)}
        self.healthy: dict[str, BackendUnavailable | None] = {}
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
        """Check once per device (without recording autograd) that the kernel builds
        and runs; a build failure raises `BackendUnavailable` (the registry records it,
        warns and falls back), anything else (out-of-memory, a bug) propagates."""
        dev = str(_device(args))
        if dev not in self.healthy:
            try:
                with torch.no_grad():
                    self.impls["kernel"](
                        *[a.detach() if isinstance(a, Tensor) else a for a in args]
                    )
                self.healthy[dev] = None
            except Exception as err:  # noqa: BLE001
                self.healthy[dev] = unavailable(self.op_name, err, dev)
        if self.healthy[dev] is not None:
            raise self.healthy[dev]

    def _lookup(self, args, qk) -> Callable:
        if self.pinned is not None:
            fn = self.impls[self.pinned]
        elif torch.compiler.is_compiling():
            fn = self.impls["kernel"]
        else:
            self._probe(args)
            row = table(_device(args), self.name).get(self.problem(args))
            if row is None and get_mode() != "off":
                self.pending[qk] = [_spec(a) for a in args]
            fn = self.impls[row["impl"] if row is not None else "kernel"]
        self.bound[qk] = fn
        return fn

    def __call__(self, *args):
        qk = _quick_key(args)
        fn = self.bound.get(qk)
        if fn is None:
            fn = self._lookup(args, qk)
        return fn(*args)

    def decide(self, *args) -> str:
        """Measure this problem (or read its stored result), store and bind the winner.
        Where the tuner may not time an autograd launch (inference mode, a checkpointed
        forward, a backward) the kernel is returned unbound and the problem stays queued.
        """
        device = _device(args)
        key = self.problem(args)
        cfg = select(
            self.name,
            device,
            key,
            lambda: [KERNEL, COMPILE],
            lambda c: _step(self.impls[c["impl"]], args),
            shortlist=2,
            errors=BUILD_ERRORS,
            autograd=True,
            timing="events",
        )
        self.choice = cfg["impl"]
        if decided(self.name, device, key):
            qk = _quick_key(args)
            self.bound[qk] = self.impls[self.choice]
            self.pending.pop(qk, None)
        return self.choice

    def bind(self, choice: str) -> None:
        """Pin `choice` for every problem of this op."""
        self.pinned = choice
        self.choice = choice
        self.bound.clear()
        self.pending.clear()

    def retune(self) -> int:
        """Measure every queued problem; returns how many were decided."""
        done = 0
        for qk, specs in list(self.pending.items()):
            args = [_rebuild(s) for s in specs]
            try:
                self.decide(*args)
            except BackendUnavailable:
                self.pending.pop(qk, None)
                continue
            if decided(self.name, _device(args), self.problem(args)):
                self.bound[qk] = self.impls[self.choice]
                self.pending.pop(qk, None)
                done += 1
        return done


def raced(backend: str, op_name: str, kernel_fn: Callable, key_fn: Callable) -> RacedOp:
    return RacedOp(backend, op_name, kernel_fn, key_fn)


def reset() -> None:
    """Forget every binding and queued problem (tuner reset, broadcast, tests)."""
    for op in list(_OPS):
        op.bound.clear()
        op.pending.clear()
        op.pinned = None
        op.choice = None


on_reset(reset)


def retune_pending() -> int:
    """Decide every queued problem of every raced op (call between steps). Nothing is
    measured where the tuner may not time an autograd launch; problems stay queued."""
    if not timing_allowed(autograd=True):
        return 0
    with torch.enable_grad():
        return sum(op.retune() for op in list(_OPS))


def shape_key(*args) -> str:
    """Problem key: tensor shapes bucketed on the leading (token) dim, dtypes, scalars."""
    parts = []
    for a in args:
        if isinstance(a, Tensor):
            dims = list(a.shape)
            if dims:
                lead = 256
                while lead < dims[0]:
                    lead *= 2
                dims[0] = lead
            parts.append(f"{'x'.join(map(str, dims))}{str(a.dtype)[6:]}")
        elif a is None:
            parts.append("none")
        elif isinstance(a, float):
            parts.append("r1" if a == 1.0 else "rx")
        else:
            parts.append(str(a))
    return "|".join(parts)
