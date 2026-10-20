"""`trim_heap`, model-free: C heap chunks freed below a live one stay resident until it
runs, and it returns them to the OS (glibc)."""

import ctypes
import os
import sys

import pytest

from kohakumoss_common.hostmem import trim_heap
from tests._util.negative import must_fail

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc heap"),
]
MIB = 2**20
CHUNK = 64 * 1024
COUNT = 2048


def _rss() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def released_by(trim) -> int:
    """Resident bytes `trim()` returns after freeing 128 MiB of 64 KiB heap chunks (below
    the mmap threshold) that sit under a live chunk."""
    libc = ctypes.CDLL(None)
    libc.malloc.restype = ctypes.c_void_p
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.free.argtypes = [ctypes.c_void_p]
    chunks = [libc.malloc(CHUNK) for _ in range(COUNT)]
    for p in chunks:
        ctypes.memset(p, 1, CHUNK)
    top = libc.malloc(CHUNK)
    ctypes.memset(top, 1, CHUNK)
    for p in chunks:
        libc.free(p)
    before = _rss()
    trim()
    released = before - _rss()
    libc.free(top)
    return released


def assert_returned(trim) -> None:
    released = released_by(trim)
    assert released > COUNT * CHUNK // 2, released / MIB


def test_trim_returns_the_freed_heap():
    assert_returned(trim_heap)
    must_fail(lambda: assert_returned(lambda: None), "freed heap kept resident")
