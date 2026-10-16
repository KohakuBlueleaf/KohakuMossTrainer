"""Op registry: backends register implementations; callers resolve once at build time.

An implementation is a callable with the op's eager signature that supports autograd
and torch.compile. `supports(query)` returns None when it can serve the query, or the
reason it cannot. `resolve` walks the backend priority for the query's arch.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

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


def register(
    op: str,
    backend: str,
    fn: Callable,
    supports: Callable[[OpQuery], str | None] | None = None,
) -> None:
    """Register `fn` as `backend`'s implementation of `op`."""
    _STATE.impls.setdefault(op, {})[backend] = Impl(backend, fn, supports)


def mark_unavailable(backend: str, reason: str) -> None:
    """Record why a whole backend failed to load."""
    _STATE.unavailable[backend] = reason


def set_priority(arch_name: str, order: tuple[str, ...] | list[str]) -> None:
    """Override the backend order for one arch entry."""
    _STATE.priority[arch_name] = tuple(order)


def available(op: str, device: torch.device | str, dtype: torch.dtype, **static):
    """`{backend: None | reason}` for `op` on `device` (the doctor's view)."""
    query = OpQuery(op, arch_of(device), dtype, tuple(sorted(static.items())))
    out: dict[str, str | None] = {}
    for backend in BACKENDS:
        if backend in _STATE.unavailable:
            out[backend] = f"backend unavailable: {_STATE.unavailable[backend]}"
            continue
        impl = _STATE.impls.get(op, {}).get(backend)
        out[backend] = "not implemented" if impl is None else impl.reason(query)
    return out


def resolve(
    op: str,
    device: torch.device | str,
    dtype: torch.dtype,
    backend: str | None = None,
    **static,
) -> Callable:
    """The implementation of `op` for (device arch, dtype, static), resolved once."""
    query = OpQuery(op, arch_of(device), dtype, tuple(sorted(static.items())))
    order = (backend,) if backend else _STATE.priority[query.arch.name]
    reasons = []
    for name in order:
        if name in _STATE.unavailable:
            reasons.append(f"{name}: unavailable ({_STATE.unavailable[name]})")
            continue
        impl = _STATE.impls.get(op, {}).get(name)
        if impl is None:
            reasons.append(f"{name}: not implemented")
            continue
        why = impl.reason(query)
        if why is None:
            return impl.fn
        reasons.append(f"{name}: {why}")
    raise NoImplementationError(
        f"{op} on {query.arch.name}/{dtype}: " + "; ".join(reasons)
    )


def load_backends() -> None:
    """Import every backend package; a failed import marks that backend unavailable."""
    for name in BACKENDS:
        try:
            importlib.import_module(f"kohakumoss_kernels.backends.{name}")
        except ImportError as err:
            mark_unavailable(name, str(err))
