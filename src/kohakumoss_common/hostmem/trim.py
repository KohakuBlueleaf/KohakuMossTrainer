"""Return the C heap's free pages to the OS (glibc `malloc_trim`). See
docs/internals/offload.md#host-heap."""

import ctypes
import ctypes.util
import sys
from functools import cache


@cache
def _malloc_trim():
    """glibc's `malloc_trim`, or None where the C library has none."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        trim = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6").malloc_trim
    except (OSError, AttributeError):
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


def trim_heap() -> bool:
    """Release the free pages of every C heap arena to the OS; False where the C
    library cannot (not glibc), a no-op there."""
    trim = _malloc_trim()
    if trim is None:
        return False
    trim(0)
    return True
