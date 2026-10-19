"""Host memory a process may plan for now: the tightest of the system's available memory
and the process's cgroup room (Linux), or the available physical memory (Windows).
The parsers are pure functions of the files' text. See
docs/internals/offload.md#host-room."""

import ctypes
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROC = Path("/proc")
CGROUP_ROOT = Path("/sys/fs/cgroup")
UNLIMITED = "max"
V1_UNLIMITED = 1 << 62
HEADROOM = 0.9


@dataclass(frozen=True)
class HostRoom:
    """`free`: bytes the host has room for now (None: no bound known); `bounds`: each
    bound by source ("meminfo", "cgroup", "windows"); `source`: the one that wins."""

    free: int | None
    bounds: dict[str, int] = field(default_factory=dict)

    @property
    def source(self) -> str:
        if not self.bounds:
            return "none"
        return min(self.bounds, key=self.bounds.get)


def meminfo_available(text: str) -> int | None:
    """MemAvailable of /proc/meminfo text, in bytes (None when absent)."""
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name == "MemAvailable":
            return int(rest.split()[0]) * 1024
    return None


def stat_value(text: str, key: str) -> int:
    """One counter of a cgroup memory.stat text (0 when absent)."""
    for line in text.splitlines():
        name, _, value = line.partition(" ")
        if name == key:
            return int(value)
    return 0


def cgroup2_room(limit: str, current: str, stat: str) -> int | None:
    """Room under a cgroup v2 memory.max: limit - usage + its reclaimable inactive file
    pages (None for an unlimited cgroup)."""
    limit = limit.strip()
    if limit == UNLIMITED:
        return None
    used = int(current.strip()) - stat_value(stat, "inactive_file")
    return max(int(limit) - used, 0)


def cgroup1_room(limit: str, usage: str, stat: str) -> int | None:
    """Room under a cgroup v1 memory.limit_in_bytes, as `cgroup2_room` (None when the
    limit is the kernel's unlimited value)."""
    cap = int(limit.strip())
    if cap >= V1_UNLIMITED:
        return None
    used = int(usage.strip()) - stat_value(stat, "total_inactive_file")
    return max(cap - used, 0)


def cgroup_paths(proc_cgroup: str, root: Path = CGROUP_ROOT) -> tuple[Path, int] | None:
    """(the process's memory cgroup dir, cgroup version) from /proc/self/cgroup text."""
    for line in proc_cgroup.splitlines():
        hier, controllers, path = line.split(":", 2)
        rel = path.lstrip("/")
        if hier == "0" and controllers == "":
            return root / rel, 2
        if "memory" in controllers.split(","):
            return root / "memory" / rel, 1
    return None


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def cgroup_room(proc: Path = PROC, root: Path = CGROUP_ROOT) -> int | None:
    """The tightest room of the process's memory cgroup and its ancestors (None: no
    limit anywhere)."""
    text = _read(proc / "self" / "cgroup")
    found = cgroup_paths(text, root) if text else None
    if found is None:
        return None
    path, version = found
    files = (
        ("memory.max", "memory.current", "memory.stat")
        if version == 2
        else ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.stat")
    )
    room = cgroup2_room if version == 2 else cgroup1_room
    stop = root if version == 2 else root / "memory"
    best = None
    while True:
        texts = [_read(path / name) for name in files]
        if all(t is not None for t in texts):
            here = room(*texts)
            if here is not None:
                best = here if best is None else min(best, here)
        if path == stop or path.parent == path:
            return best
        path = path.parent


def linux_room(proc: Path = PROC, root: Path = CGROUP_ROOT) -> HostRoom:
    bounds = {}
    text = _read(proc / "meminfo")
    avail = meminfo_available(text) if text else None
    if avail is not None:
        bounds["meminfo"] = avail
    cg = cgroup_room(proc, root)
    if cg is not None:
        bounds["cgroup"] = cg
    return HostRoom(min(bounds.values()) if bounds else None, bounds)


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def windows_room(status: _MemoryStatusEx | None = None) -> HostRoom:
    """Windows: the available physical memory of GlobalMemoryStatusEx (standby pages
    included); `status` a filled structure (default: the call)."""
    if status is None:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return HostRoom(None)
    free = int(status.ullAvailPhys)
    return HostRoom(free, {"windows": free})


def host_room() -> HostRoom:
    """The host's room now, on this platform."""
    match sys.platform:
        case "win32":
            return windows_room()
        case "linux":
            return linux_room()
    return HostRoom(None)


def host_fits(nbytes: int, room: HostRoom | None = None) -> bool:
    """Whether `nbytes` more fit in HEADROOM x the host's room now (`room`, default
    `host_room()`); True when no bound is known."""
    room = host_room() if room is None else room
    return room.free is None or int(nbytes) <= int(room.free * HEADROOM)
