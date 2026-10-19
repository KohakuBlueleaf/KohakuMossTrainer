"""Op registry: backends register implementations; callers `resolve` once at build time
to a `Bound` op that falls back along the arch's backend priority. See
docs/internals/kernel-dev.md#registry.
"""

import gc
import importlib
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from kohakumoss_common.log import debug_once, get_logger, info_once, warning_once
from kohakumoss_kernels import compile_setup, status
from kohakumoss_kernels.dtypes import short
from kohakumoss_kernels.errors import (
    LAUNCH_ERRORS,
    BackendUnavailable,
    InputError,
    is_sticky_error,
    unavailable,
)
from kohakumoss_kernels.hardware.arch import Arch, arch_of
from kohakumoss_kernels.hardware.current import saved_tensor_hooks_active
from kohakumoss_kernels.recording import RECORDS, on_change, record

BACKENDS = ("tilelang", "triton", "compile", "eager")
log = get_logger(__name__)

DEFAULT_PRIORITY: dict[str, tuple[str, ...]] = {
    "sm70": ("tilelang", "triton", "compile", "eager"),
    "sm75": ("tilelang", "triton", "compile", "eager"),
    "sm80": ("triton", "tilelang", "compile", "eager"),
    "sm86": ("triton", "tilelang", "compile", "eager"),
    "sm89": ("triton", "tilelang", "compile", "eager"),
    "sm90": ("triton", "tilelang", "compile", "eager"),
    "sm100": ("triton", "tilelang", "compile", "eager"),
    "sm120": ("triton", "tilelang", "compile", "eager"),
    "xpu": ("triton", "compile", "eager"),
    "cpu": ("eager",),
}
CAUGHT = (BackendUnavailable, *LAUNCH_ERRORS)
_LOAD_LOCK = threading.RLock()
_LOADED: list[bool] = []
_LOADING: list[bool] = []


@dataclass(frozen=True)
class OpQuery:
    """What a caller asks for: op name, arch, compute dtype and static options."""

    op: str
    arch: Arch
    dtype: torch.dtype
    static: tuple[tuple[str, object], ...] = ()

    def get(self, key: str, default=None):
        return dict(self.static).get(key, default)


@dataclass
class Impl:
    """One backend's implementation of one op."""

    backend: str
    fn: Callable
    supports: Callable[[OpQuery], str | None] | None = None

    def reason(self, query: OpQuery) -> str | None:
        return None if self.supports is None else self.supports(query)


@dataclass
class _State:
    impls: dict[str, dict[str, Impl]] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
    priority: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_PRIORITY)
    )


_STATE = _State()


class NoImplementationError(RuntimeError):
    """No registered backend can serve the query; the message lists every reason."""


# the doctor's view of recorded failures
failures = status.failures
recorded_failures = status.recorded_failures


_LIVE: "weakref.WeakSet[Bound]" = weakref.WeakSet()
# ops whose guarded backward succeeded, unguarded at the next `settle_guards`
PROVEN: "weakref.WeakSet[Bound]" = weakref.WeakSet()


class Bound:
    """A resolved op: calls its implementation directly; a build / launch failure
    (forward, or backward until one has succeeded) records it, logs a WARNING and
    rebinds this op and every live `Bound` of the same query on the failed backend to
    the next backend of `order`. The call path (recording, guarded, plain) is bound
    when the recording state or `guard` changes. See
    docs/internals/kernel-dev.md#fallback."""

    def __init__(self, query: OpQuery, device, order: tuple[str, ...], backend: str):
        self.op = query.op
        self.query = query
        self.device = device
        self.order = order
        self.backend = backend
        self.fn = _STATE.impls[query.op][backend].fn
        self.guard = len(order) > 1
        _LIVE.add(self)

    @property
    def guard(self) -> bool:
        """Whether calls that record a plain autograd graph run through `_Guarded`."""
        return self._guard

    @guard.setter
    def guard(self, on: bool) -> None:
        self._guard = on
        self.rebind()

    def rebind(self) -> None:
        """Bind the call path (a class function) for the current recording state and
        `guard`."""
        if RECORDS:
            self._call = Bound._record
        elif self._guard:
            self._call = Bound._guarded
        else:
            self._call = Bound._plain

    def __call__(self, *args, **kwargs):
        return self._call(self, *args, **kwargs)

    def _plain(self, *args, **kwargs):
        try:
            return self.fn(*args, **kwargs)
        except CAUGHT as err:
            self.fall_back(err, args, kwargs)
        return self(*args, **kwargs)

    def _guarded(self, *args, **kwargs):
        try:
            if _records_grad(args):
                return _Guarded.apply(self, kwargs, *args)
            return self.fn(*args, **kwargs)
        except CAUGHT as err:
            self.fall_back(err, args, kwargs)
        return self(*args, **kwargs)

    def __getattr__(self, name: str):
        if name in ("fn", "_guard", "_call"):
            raise AttributeError(name)
        return getattr(self.fn, name)

    def _record(self, *args, **kwargs):
        """Inside `recording()`: snapshot this call and answer with the eager reference,
        so nothing is built or tuned."""
        record(self, args, kwargs)
        eager = _STATE.impls[self.op].get("eager")
        return (eager.fn if eager is not None else self.fn)(*args, **kwargs)

    def fall_back(self, err: BaseException, args=None, kwargs=None) -> None:
        """Record that the current backend cannot run this op and rebind to the next
        one (re-raises `err` when none is left or the backend was pinned). With the
        failing call's `args`, raises `InputError` instead when the eager reference
        rejects them too: nothing is recorded and the op stays bound."""
        if is_sticky_error(err, self.device):
            raise err
        if args is not None:
            self._check_inputs(err, args, kwargs or {})
        err = unavailable(self.op, err, self.backend)
        arch = self.query.arch.name
        status.record(
            self.op, self.backend, arch, err, self.query.dtype, self.query.static
        )
        rest = self.order[self.order.index(self.backend) + 1 :]
        if not rest:
            raise err
        try:
            name, _ = _first(self.query, self.device, rest)
        except NoImplementationError as none_left:
            raise none_left from err
        failed = self.backend
        siblings = [
            op
            for op in list(_LIVE)
            if op is not self
            and op.backend == failed
            and op.query == self.query
            and op.order == self.order
        ]
        for op in (self, *siblings):
            op._switch(name)
        log.warning(
            f"{self.op} on {arch}/{short(self.query.dtype)}: backend {failed} failed, "
            f"using {name} ({len(siblings) + 1} bound ops): {err}"[:4000]
        )

    def _switch(self, name: str) -> None:
        self.backend = name
        self.fn = _STATE.impls[self.op][name].fn
        self.guard = name != self.order[-1]

    def _check_inputs(self, err: BaseException, args, kwargs) -> None:
        """Raise `InputError` when the eager reference also fails on `args`."""
        eager = _STATE.impls[self.op].get("eager")
        if eager is None or self.backend == "eager":
            return
        try:
            with torch.no_grad():
                eager.fn(*_leaves(args), **kwargs)
        except torch.OutOfMemoryError:
            return
        except Exception as rejected:  # noqa: BLE001
            raise InputError(self.op, self.backend, err, rejected) from err


def _rebind_live() -> None:
    for op in list(_LIVE):
        op.rebind()


on_change(_rebind_live)


def _records_grad(args) -> bool:
    """A call that records a plain autograd graph: grad on, not traced, no saved-tensor
    hooks (checkpoint / offload keep their own graph semantics), a tensor needing grad.
    """
    return (
        torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and not saved_tensor_hooks_active()
        and any(isinstance(a, torch.Tensor) and a.requires_grad for a in args)
    )


def _leaves(args) -> list:
    return [
        a.detach().requires_grad_(a.requires_grad) if isinstance(a, torch.Tensor) else a
        for a in args
    ]


def _grads(fn, kwargs, leaves, grads) -> tuple:
    """Gradients of fn(*leaves) for the upstream `grads`, per leaf (None if unused)."""
    with torch.enable_grad():
        out = fn(*leaves, **kwargs)
    outs = out if isinstance(out, tuple) else (out,)
    return _backprop(outs, leaves, grads)


def _backprop(outs, leaves, grads) -> tuple:
    need = [
        i
        for i, a in enumerate(leaves)
        if isinstance(a, torch.Tensor) and a.requires_grad
    ]
    pairs = [(o, g) for o, g in zip(outs, grads) if g is not None and o.requires_grad]
    got = torch.autograd.grad(
        [o for o, _ in pairs],
        [leaves[i] for i in need],
        [g for _, g in pairs],
        allow_unused=True,
    )
    result = [None] * len(leaves)
    for i, g in zip(need, got):
        result[i] = g
    return tuple(result)


class _Guarded(torch.autograd.Function):
    """Runs a `Bound` op's implementation on detached leaves and its backward through
    autograd.grad; a backend failure in that backward falls the op back and recomputes
    forward + backward with the next backend from the saved inputs (version-checked).
    The backward releases the forward's leaves, outputs and arguments, so the graph
    keeps none of them after it; a good backward marks the op proven (`settle_guards`
    then binds its plain path)."""

    @staticmethod
    def forward(ctx, op, kwargs, *args):
        leaves = _leaves(args)
        with torch.enable_grad():
            out = op.fn(*leaves, **kwargs)
        ctx.save_for_backward(*(a for a in args if isinstance(a, torch.Tensor)))
        ctx.slots = [None if isinstance(a, torch.Tensor) else (a,) for a in args]
        ctx.op, ctx.kwargs, ctx.leaves = op, kwargs, leaves
        ctx.outs = out if isinstance(out, tuple) else (out,)
        if isinstance(out, tuple):
            return tuple(o.detach() for o in out)
        return out.detach()

    @staticmethod
    def backward(ctx, *grads):
        op = ctx.op
        try:
            got = _backprop(ctx.outs, ctx.leaves, grads)
            PROVEN.add(op)
        except CAUGHT as err:
            saved = iter(ctx.saved_tensors)
            args = [next(saved) if s is None else s[0] for s in ctx.slots]
            op.fall_back(err, args, ctx.kwargs)
            got = _grads(op.fn, ctx.kwargs, _leaves(args), grads)
        finally:
            ctx.op = ctx.kwargs = ctx.leaves = ctx.outs = ctx.slots = None
        return (None, None, *got)


def settle_guards() -> int:
    """Bind the plain call path of every op whose guarded backward succeeded (at a step
    boundary, so one step runs one call path per op); returns how many."""
    ops = list(PROVEN)
    PROVEN.clear()
    for op in ops:
        op.guard = False
    return len(ops)


def exercise(op: Bound, run: Callable[[Bound], object]) -> object:
    """`run(op)` (the prewarm: a forward + backward on scratch inputs) as build time
    (`status.build_phase`); a backend failure raised anywhere in it, the backward
    included, falls `op` back and runs it again; a good backward unguards `op` there
    (no step is running)."""
    with status.build_phase():
        while True:
            try:
                out = run(op)
            except CAUGHT as err:
                op.fall_back(err)
                continue
            if op in PROVEN:
                PROVEN.discard(op)
                op.guard = False
            return out


def register(
    op: str,
    backend: str,
    fn: Callable,
    supports: Callable[[OpQuery], str | None] | None = None,
) -> None:
    """Register `fn` as `backend`'s implementation of `op`."""
    _STATE.impls.setdefault(op, {})[backend] = Impl(backend, fn, supports)


def mark_unavailable(backend: str, reason: str) -> None:
    """Record why a whole backend cannot be used in this process."""
    _STATE.unavailable[backend] = reason


def set_priority(arch_name: str, order: tuple[str, ...] | list[str]) -> None:
    """Override the backend order for one arch entry."""
    _STATE.priority[arch_name] = tuple(order)


def reset_priority() -> None:
    """Every arch entry back to its default backend order (`set_priority` undone)."""
    _STATE.priority.clear()
    _STATE.priority.update(DEFAULT_PRIORITY)


def _reason(name: str, query: OpQuery, device) -> str | None:
    """Why backend `name` cannot serve `query` on `device`, or None."""
    if name in _STATE.unavailable:
        return f"unavailable ({_STATE.unavailable[name]})"
    impl = _STATE.impls.get(query.op, {}).get(name)
    if impl is None:
        return "not implemented"
    why = impl.reason(query)
    if why is not None:
        return why
    failed = status.failed(query.op, name, query.arch.name, query.dtype, query.static)
    if failed is not None:
        return f"failed here ({failed[:300]})"
    why = status.probe(name, device)
    return None if why is None else f"cannot build on {device} ({why[:300]})"


def available(op: str, device: torch.device | str, dtype: torch.dtype, **static):
    """`{backend: None | reason}` for `op` on `device` (the doctor's view)."""
    load_backends()
    query = OpQuery(op, arch_of(device), dtype, tuple(sorted(static.items())))
    return {b: _reason(b, query, device) for b in BACKENDS}


def _first(query: OpQuery, device, order: tuple[str, ...]) -> tuple[str, Impl]:
    """The first backend of `order` that can serve `query` on `device`."""
    reasons = []
    for name in order:
        why = _reason(name, query, device)
        if why is None:
            return name, _STATE.impls[query.op][name]
        if why.startswith("failed here"):
            warning_once(
                log.name,
                f"{query.op} on {query.arch.name}/{short(query.dtype)}: backend "
                f"{name} skipped, {why}",
            )
        reasons.append(f"{name}: {why}")
    raise NoImplementationError(
        f"{query.op} on {query.arch.name}/{query.dtype}: " + "; ".join(reasons)
    )


def resolve(
    op: str,
    device: torch.device | str,
    dtype: torch.dtype,
    backend: str | None = None,
    **static,
) -> Bound:
    """The implementation of `op` for (device, dtype, static), resolved once; a pinned
    `backend` gets no fallback. The first call loads the backends (`load_backends`)."""
    load_backends()
    query = OpQuery(op, arch_of(device), dtype, tuple(sorted(static.items())))
    order = (backend,) if backend else _STATE.priority[query.arch.name]
    name, _ = _first(query, device, order)
    once = debug_once if torch.device(device).type == "cpu" else info_once
    once(log.name, f"{op} on {query.arch.name}/{short(dtype)}: backend {name}")
    return Bound(query, device, order, name)


def load_backends() -> None:
    """Once per process, on the first kernel build: apply the compile settings
    (`compile_setup`: cache key tag, never-quiesced compile pool) and import every
    backend package (each sets up the host toolchain); a failed import
    marks that backend unavailable; then one cyclic gc pass, so no import-time cycle
    holds the caller's frames. See docs/internals/kernel-dev.md#lazy-backends."""
    if _LOADED:
        return
    with _LOAD_LOCK:
        if _LOADED or _LOADING:
            return
        _LOADING.append(True)
        try:
            compile_setup.apply()
            for name in BACKENDS:
                try:
                    importlib.import_module(f"kohakumoss_kernels.backends.{name}")
                except ImportError as err:
                    mark_unavailable(name, str(err))
            gc.collect()
        finally:
            _LOADING.clear()
        _LOADED.append(True)
