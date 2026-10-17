"""Backend health per device and recorded backend failures.

A backend registers a smoke check (`register_check`); `healthy(backend, device)` runs it
once per process and device. Failures of an op on a backend (`record`) are kept in
memory and on disk (`<kernel cache>/failures/backends.json`) under the environment
fingerprint (package versions, toolchain), and are honoured by later processes with the
same fingerprint. See docs/internals/kernel-dev.md.
"""

import hashlib
import importlib.metadata
import sys
import time
import warnings
from collections.abc import Callable
from functools import cache

import torch

from kohakumoss_kernels.errors import describe, is_build_error
from kohakumoss_kernels.hardware.device import cache_dir, read_json, write_json
from kohakumoss_kernels.hardware.toolchain import ensure_toolchain

VERSIONED = (
    "torch",
    "triton",
    "triton-windows",
    "triton-xpu",
    "tilelang",
    "kohakumoss",
)
_CHECKS: dict[str, Callable[[torch.device], None]] = {}
_HEALTH: dict[tuple[str, str], str | None] = {}
_FAILED: dict[tuple[str, str, str], str] = {}
_DISK: dict[str, dict] = {}
_LOADED: list[bool] = []


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "-"


@cache
def fingerprint() -> str:
    """Hash of the package versions and the toolchain a failure was recorded under."""
    tc = ensure_toolchain()
    parts = [f"{d}={_version(d)}" for d in VERSIONED]
    parts += [sys.version.split()[0], tc.source, str(tc.cxx), str(tc.msvc_version)]
    parts.append(tc.sources)
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def _path():
    return cache_dir() / "failures" / "backends.json"


def _disk() -> dict[str, dict]:
    if not _LOADED:
        _DISK.update(read_json(_path()) or {})
        _LOADED.append(True)
    return _DISK


def _key(op: str, backend: str, arch: str) -> str:
    return f"{op}|{backend}|{arch}"


def register_check(backend: str, check: Callable[[torch.device], None]) -> None:
    """`check(device)` builds and launches a minimal kernel of `backend` on `device`."""
    _CHECKS[backend] = check


def healthy(backend: str, device: torch.device) -> str | None:
    """None when `backend` builds and launches on `device` (checked once), else why."""
    device = torch.device(device)
    if device.type != "cpu" and device.index is None:
        device = torch.device(device.type, 0)
    ident = (backend, str(device))
    if ident not in _HEALTH:
        check = _CHECKS.get(backend)
        why = None
        if check is not None:
            try:
                check(device)
            except Exception as err:  # noqa: BLE001
                if not is_build_error(err):
                    raise
                why = describe(err)
                warnings.warn(
                    f"kohakumoss_kernels: backend {backend} cannot build on {device}: "
                    f"{why}"[:4000],
                    stacklevel=2,
                )
        _HEALTH[ident] = why
    return _HEALTH[ident]


def record(op: str, backend: str, arch: str, err: BaseException) -> str:
    """Record that `backend` cannot run `op` on `arch` (memory and disk)."""
    text = describe(err)
    _FAILED[(op, backend, arch)] = text
    fp = fingerprint()
    disk = {k: v for k, v in _disk().items() if v.get("fingerprint") == fp}
    disk[_key(op, backend, arch)] = {
        "error": text,
        "time": time.time(),
        "fingerprint": fp,
    }
    _DISK.clear()
    _DISK.update(disk)
    try:
        write_json(_path(), disk)
    except OSError:
        pass
    return text


def failed(op: str, backend: str, arch: str) -> str | None:
    """Why `backend` failed `op` on `arch` here or in an earlier process with the same
    fingerprint; None when it did not."""
    text = _FAILED.get((op, backend, arch))
    if text is not None:
        return text
    row = _disk().get(_key(op, backend, arch))
    if row is not None and row.get("fingerprint") == fingerprint():
        return row["error"]
    return None


def failures() -> dict[str, str]:
    """`{"op|backend|arch": error}` recorded in this process."""
    return {_key(*k): v for k, v in _FAILED.items()}


def recorded_failures() -> dict[str, dict]:
    """Failures on disk recorded under the current fingerprint."""
    fp = fingerprint()
    return {k: v for k, v in _disk().items() if v.get("fingerprint") == fp}


def health_report() -> dict[str, str | None]:
    """`{"backend@device": None | why}` of every check run in this process."""
    return {f"{b}@{d}": why for (b, d), why in _HEALTH.items()}


def clear_failures() -> None:
    """Forget every recorded failure (memory and disk) and every health result."""
    _FAILED.clear()
    _HEALTH.clear()
    _DISK.clear()
    _LOADED[:] = [True]
    try:
        write_json(_path(), {})
    except OSError:
        pass
