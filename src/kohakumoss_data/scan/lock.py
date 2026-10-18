"""Single-writer coordination for a vault file: an exclusive lock file `<path>.lock`
(writer pid and host) and a completion marker `<path>.ready`.
See docs/concepts/data.md#the-catalog-cache."""

import ctypes
import json
import os
import socket
import time
from ctypes import wintypes

from kohakumoss_common.io import read_json, write_json_atomic

LOCK, READY = ".lock", ".ready"
HOST = socket.gethostname()
STILL_ACTIVE = 259
QUERY = 0x1000
UNREADABLE_STALE = 60.0


def _alive_posix(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kernel32():
    """kernel32 with the prototypes `_alive_windows` calls (a HANDLE is pointer-sized:
    the default int restype would truncate it on 64-bit Windows)."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def _alive_windows(pid: int) -> bool:
    kernel = KERNEL32
    handle = kernel.OpenProcess(QUERY, False, pid)
    if not handle:
        return False
    code = wintypes.DWORD()
    try:
        ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        return bool(ok) and code.value == STILL_ACTIVE
    finally:
        kernel.CloseHandle(handle)


KERNEL32 = kernel32() if os.name == "nt" else None
pid_alive = _alive_windows if os.name == "nt" else _alive_posix


def stale(path: str) -> bool:
    """The lock of `path` is held by a dead process on this host (a lock from another
    host is never stale), or is unreadable and older than UNREADABLE_STALE seconds."""
    owner = read_json(path + LOCK)
    if owner is None:
        return os.path.exists(path + LOCK) and _age(path + LOCK) > UNREADABLE_STALE
    return owner.get("host") == HOST and not pid_alive(int(owner.get("pid", -1)))


def _age(path: str) -> float:
    try:
        return time.time() - os.path.getmtime(path)
    except OSError:
        return 0.0


def acquire(path: str) -> bool:
    """Take the write lock of `path`; False when another live process holds it."""
    for _ in range(2):
        try:
            fd = os.open(path + LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not stale(path):
                return False
            release(path)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"pid": os.getpid(), "host": HOST}, fh)
        return True
    return False


def release(path: str) -> None:
    """Remove the write lock of `path`."""
    try:
        os.remove(path + LOCK)
    except FileNotFoundError:
        pass


def held(path: str) -> bool:
    """A live process holds the write lock of `path`."""
    return os.path.exists(path + LOCK) and not stale(path)


def write_ready(path: str, info: dict) -> None:
    """Publish the completion marker atomically."""
    marker = {"generation": info["generation"], "rows": info["rows"]}
    write_json_atomic(path + READY, marker, indent=None)


def read_ready(path: str) -> dict | None:
    return read_json(path + READY)


def clear_ready(path: str) -> None:
    try:
        os.remove(path + READY)
    except FileNotFoundError:
        pass


def wait_ready(path: str, timeout: float, poll: float = 0.25) -> dict:
    """Block until no live writer holds `path` and its marker exists; the marker."""
    deadline = time.monotonic() + timeout
    while True:
        marker = read_ready(path)
        if marker is not None and not held(path):
            return marker
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"{path}: no completed catalog after {timeout:.0f} s (writer lock "
                f"{'held' if held(path) else 'free'}); delete {path + LOCK} if no "
                "writer is running"
            )
        time.sleep(poll)
