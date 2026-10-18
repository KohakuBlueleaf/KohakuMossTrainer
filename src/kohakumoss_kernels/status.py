"""Backend status per device (`register_check` / `probe`) and recorded backend
failures (`record` / `failed`, keyed by op, backend, arch, dtype, static options and,
for a failure of one problem, that problem). Only build / toolchain failures met at
build time (`build_phase`: the prewarm) go to disk, per environment fingerprint; a
failure in a live call stays in this process. See docs/internals/kernel-dev.md#fallback.
"""

import hashlib
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import cache

import torch

from kohakumoss_common.log import get_logger
from kohakumoss_kernels.dtypes import short
from kohakumoss_kernels.errors import describe, is_build_error, is_toolchain_error
from kohakumoss_kernels.hardware.device import cache_dir, read_json, save_json
from kohakumoss_kernels.hardware.toolchain import ensure_toolchain, version

VERSIONED = (
    "torch",
    "triton",
    "triton-windows",
    "triton-xpu",
    "tilelang",
    "kohakumoss",
)
_CHECKS: dict[str, Callable[[torch.device], None]] = {}
_PROBED: dict[tuple[str, str], str | None] = {}
_FAILED: dict[str, str] = {}
_DISK: dict[str, dict] = {}
_LOADED: list[bool] = []
_BUILD: list[bool] = []
log = get_logger(__name__)


@contextmanager
def build_phase():
    """Inside, a recorded build / toolchain failure is written to disk (the prewarm)."""
    _BUILD.append(True)
    try:
        yield
    finally:
        _BUILD.pop()


@cache
def fingerprint() -> str:
    """Hash of the package versions and the toolchain a failure was recorded under."""
    tc = ensure_toolchain()
    parts = [f"{d}={version(d) or '-'}" for d in VERSIONED]
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


def failure_key(
    op: str,
    backend: str,
    arch: str,
    dtype: torch.dtype | None = None,
    static: tuple = (),
    problem: str | None = None,
) -> str:
    """`op|backend|arch|dtype|static[|problem]`: what a recorded failure covers."""
    options = ",".join(f"{k}={v}" for k, v in static) or "-"
    parts = [op, backend, arch, "-" if dtype is None else short(dtype), options]
    return "|".join(parts if problem is None else [*parts, problem])


def register_check(backend: str, check: Callable[[torch.device], None]) -> None:
    """`check(device)` builds and launches a minimal kernel of `backend` on `device`."""
    _CHECKS[backend] = check


def probe(backend: str, device: torch.device) -> str | None:
    """None when `backend` builds and launches on `device` (checked once), else why."""
    device = torch.device(device)
    if device.type != "cpu" and device.index is None:
        device = torch.device(device.type, 0)
    ident = (backend, str(device))
    if ident not in _PROBED:
        check = _CHECKS.get(backend)
        why = None
        if check is not None:
            try:
                check(device)
            except Exception as err:  # noqa: BLE001
                if not is_build_error(err):
                    raise
                why = describe(err)
                log.warning(f"backend {backend} cannot build on {device}: {why}"[:4000])
        _PROBED[ident] = why
    return _PROBED[ident]


def record(
    op: str,
    backend: str,
    arch: str,
    err: BaseException,
    dtype: torch.dtype | None = None,
    static: tuple = (),
) -> str:
    """Record that `backend` cannot run `op` on `arch` for `dtype` / `static` (for the
    one problem `err.problem` names, when it names one): in memory always, on disk only
    for a build / toolchain cause (`is_toolchain_error`) met inside `build_phase`."""
    text = describe(err)
    key = failure_key(op, backend, arch, dtype, static, getattr(err, "problem", None))
    _FAILED[key] = text
    if not (_BUILD and is_toolchain_error(err)):
        return text
    fp = fingerprint()
    disk = {k: v for k, v in _disk().items() if v.get("fingerprint") == fp}
    disk[key] = {"error": text, "time": time.time(), "fingerprint": fp}
    _DISK.clear()
    _DISK.update(disk)
    save_json(_path(), disk)
    return text


def failed(
    op: str,
    backend: str,
    arch: str,
    dtype: torch.dtype | None = None,
    static: tuple = (),
) -> str | None:
    """Why `backend` failed `op` on `arch` for `dtype` / `static` as a whole (not one
    problem) here or in an earlier process with the same fingerprint; None when it
    did not."""
    key = failure_key(op, backend, arch, dtype, static)
    text = _FAILED.get(key)
    if text is not None:
        return text
    row = _disk().get(key)
    if row is not None and row.get("fingerprint") == fingerprint():
        return row["error"]
    return None


def failures() -> dict[str, str]:
    """`{failure_key: error}` recorded in this process."""
    return dict(_FAILED)


def recorded_failures() -> dict[str, dict]:
    """Failures on disk recorded under the current fingerprint."""
    fp = fingerprint()
    return {k: v for k, v in _disk().items() if v.get("fingerprint") == fp}


def status_report() -> dict[str, str | None]:
    """`{"backend@device": None | why}` of every probe run in this process."""
    return {f"{b}@{d}": why for (b, d), why in _PROBED.items()}


def clear_failures() -> None:
    """Forget every recorded failure (memory and disk) and every probe result."""
    _FAILED.clear()
    _PROBED.clear()
    _DISK.clear()
    _LOADED[:] = [True]
    save_json(_path(), {})


def forget_session() -> None:
    """Forget the failures and probe results this process recorded in memory (a new
    run in the same process starts from the disk record only)."""
    _FAILED.clear()
    _PROBED.clear()
