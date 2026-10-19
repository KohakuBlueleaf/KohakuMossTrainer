"""The host's room, model-free from fixture files: MemAvailable, a cgroup v2 / v1
limit minus usage plus its inactive file pages, the tightest ancestor, the Windows
available physical memory, and the headroom `host_fits` keeps."""

from pathlib import Path

import pytest

from kohakumoss_common.hostmem import available as av
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
GIB = 2**30
MEMINFO = "MemTotal: 33554432 kB\nMemFree: 1048576 kB\nMemAvailable: 13312000 kB\n"


def _tree(tmp: Path, v2: dict[str, tuple[str, str, str]]) -> tuple[Path, Path]:
    """A /proc and cgroup v2 root from fixture files: `v2` maps a cgroup path to (max,
    current, inactive_file); the process sits in the deepest one."""
    proc, root = tmp / "proc", tmp / "cg"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text(MEMINFO)
    deepest = max(v2, key=len)
    (proc / "self" / "cgroup").write_text(f"0::/{deepest}\n")
    for rel, (cap, cur, inactive) in v2.items():
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / "memory.max").write_text(cap + "\n")
        (d / "memory.current").write_text(cur + "\n")
        (d / "memory.stat").write_text(f"anon 1\ninactive_file {inactive}\n")
    return proc, root


def assert_cgroup_bound(room_fn, tmp: Path) -> None:
    proc, root = _tree(
        tmp,
        {
            "user.slice": ("max", str(20 * GIB), "0"),
            "user.slice/run.scope": (str(12 * GIB), str(10 * GIB), str(3 * GIB)),
        },
    )
    room = room_fn(proc, root)
    assert room.bounds["meminfo"] == 13312000 * 1024
    assert room.bounds.get("cgroup") == 12 * GIB - (10 * GIB - 3 * GIB)
    assert room.free == 5 * GIB and room.source == "cgroup"


def test_cgroup_v2_bounds_the_room(tmp_path):
    assert_cgroup_bound(av.linux_room, tmp_path)


def test_tightest_ancestor_wins(tmp_path):
    proc, root = _tree(
        tmp_path,
        {
            "a": (str(4 * GIB), str(3 * GIB), "0"),
            "a/b": (str(12 * GIB), str(2 * GIB), "0"),
        },
    )
    assert av.cgroup_room(proc, root) == GIB


def test_unlimited_cgroup_leaves_meminfo(tmp_path):
    proc, root = _tree(tmp_path, {"x": ("max", str(GIB), "0")})
    room = av.linux_room(proc, root)
    assert room.bounds == {"meminfo": 13312000 * 1024} and room.source == "meminfo"


def test_cgroup_v1_room():
    stat = "cache 5\ntotal_inactive_file 1073741824\n"
    assert av.cgroup1_room(str(8 * GIB), str(6 * GIB), stat) == 3 * GIB
    assert av.cgroup1_room(str(1 << 63), str(6 * GIB), stat) is None
    assert av.cgroup_paths("4:memory:/docker/abc\n", Path("/r")) == (
        Path("/r/memory/docker/abc"),
        1,
    )


def test_windows_room():
    s = av._MemoryStatusEx()
    s.ullTotalPhys, s.ullAvailPhys = 32 * GIB, 11 * GIB
    room = av.windows_room(s)
    assert room.free == 11 * GIB and room.source == "windows"


def assert_fits(fits) -> None:
    room = av.HostRoom(10 * GIB, {"cgroup": 10 * GIB})
    assert fits(9 * GIB, room)
    assert not fits(int(9.5 * GIB), room)
    assert fits(1 << 50, av.HostRoom(None))


def test_host_fits_keeps_headroom():
    assert_fits(av.host_fits)


def _no_cgroup(proc, root):
    return av.HostRoom(
        av.meminfo_available((proc / "meminfo").read_text()),
        {"meminfo": av.meminfo_available((proc / "meminfo").read_text())},
    )


def _no_reclaim(limit, current, stat):
    return int(limit) - int(current) if limit.strip() != "max" else None


def _no_headroom(nbytes, room):
    return room.free is None or nbytes <= room.free


def test_checks_fail_on_broken_readers(tmp_path, monkeypatch):
    must_fail(lambda: assert_cgroup_bound(_no_cgroup, tmp_path / "a"), "cgroup ignored")
    must_fail(lambda: assert_fits(_no_headroom), "no headroom")
    monkeypatch.setattr(av, "cgroup2_room", _no_reclaim)
    must_fail(
        lambda: assert_cgroup_bound(av.linux_room, tmp_path / "b"),
        "inactive file pages not reclaimable",
    )
