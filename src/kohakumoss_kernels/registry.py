"""Op registry: backends register implementations; callers resolve once at build time.

An implementation is a callable with the op's eager signature that supports autograd
and torch.compile. `supports(query)` returns None when it can serve the query, or the
reason it cannot. `resolve` walks the backend priority for the query's arch, skipping
backends that fail their health check on the device or have a recorded failure for the
op, and returns a `Bound` op that falls back along the priority when its implementation
cannot run. See docs/internals/kernel-dev.md.
"""

import importlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from kohakumoss_kernels import health
from kohakumoss_kernels.errors import LAUNCH_ERRORS, BackendUnavailable, unavailable
from kohakumoss_kernels.hardware.arch import Arch, arch_of

BACKENDS = ("tilelang", "triton", "compile", "eager")

DEFAULT_PRIORITY: dict[str, tuple[str, ...]] = {
    "sm70": ("tilelang", "compile", "eager"),
    "sm75": ("tilelang", "compile", "eager"),
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


failures = health.failures
recorded_failures = health.recorded_failures


class Bound:
    """A resolved op: calls its implementation directly. When the implementation
    cannot run here (`BackendUnavailable`, a Triton build / resource error) the
    failure is recorded, a warning names it, and the op rebinds to the next backend of
    `order` and runs again; a pinned backend re-raises."""

    def __init__(self, query: OpQuery, device, order: tuple[str, ...], backend: str):
        self.op = query.op
        self.query = query
        self.device = device
        self.order = order
        self.backend = backend
        self.fn = _STATE.impls[query.op][backend].fn

    def __call__(self, *args, **kwargs):
        try:
            return self.fn(*args, **kwargs)
        except CAUGHT as err:
            self.fall_back(err)
        return self(*args, **kwargs)

    def __getattr__(self, name: str):
        if name == "fn":
            raise AttributeError(name)
        return getattr(self.fn, name)

    def fall_back(self, err: BaseException) -> None:
        """Record that the current backend cannot run this op and rebind to the next
        one (re-raises `err` when none is left or the backend was pinned)."""
        err = unavailable(self.op, err, self.backend)
        arch = self.query.arch.name
        health.record(self.op, self.backend, arch, err)
        rest = self.order[self.order.index(self.backend) + 1 :]
        if not rest:
            raise err
        try:
            name, _ = _first(self.query, self.device, rest)
        except NoImplementationError as none_left:
            raise none_left from err
        warnings.warn(
            f"kohakumoss_kernels: {self.op} on {arch}: backend {self.backend} failed, "
            f"using {name}: {err}"[:4000],
            stacklevel=3,
        )
        self.backend = name
        self.fn = _STATE.impls[self.op][name].fn


def exercise(op: Bound, run: Callable[[Bound], object]) -> object:
    """`run(op)` (e.g. a forward + backward on scratch inputs); a backend failure raised
    anywhere in it, the backward included, falls `op` back and runs it again."""
    while True:
        try:
            return run(op)
        except CAUGHT as err:
            op.fall_back(err)


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
    failed = health.failed(query.op, name, query.arch.name)
    if failed is not None:
        return f"failed here ({failed[:300]})"
    sick = health.healthy(name, device)
    return None if sick is None else f"cannot build on {device} ({sick[:300]})"


def available(op: str, device: torch.device | str, dtype: torch.dtype, **static):
    """`{backend: None | reason}` for `op` on `device` (the doctor's view)."""
    query = OpQuery(op, arch_of(device), dtype, tuple(sorted(static.items())))
    return {b: _reason(b, query, device) for b in BACKENDS}


def _first(query: OpQuery, device, order: tuple[str, ...]) -> tuple[str, Impl]:
    """The first backend of `order` that can serve `query` on `device`."""
    reasons = []
    for name in order:
        why = _reason(name, query, device)
        if why is None:
            return name, _STATE.impls[query.op][name]
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
    `backend` gets no fallback."""
    query = OpQuery(op, arch_of(device), dtype, tuple(sorted(static.items())))
    order = (backend,) if backend else _STATE.priority[query.arch.name]
    name, _ = _first(query, device, order)
    return Bound(query, device, order, name)


def load_backends() -> None:
    """Import every backend package; a failed import marks that backend unavailable."""
    for name in BACKENDS:
        try:
            importlib.import_module(f"kohakumoss_kernels.backends.{name}")
        except ImportError as err:
            mark_unavailable(name, str(err))
