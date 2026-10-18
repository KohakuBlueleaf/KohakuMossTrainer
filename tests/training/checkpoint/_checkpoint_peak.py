"""Child process for test_checkpoint_stream: saves a 256 MB state of strided tensors
(each needs a host copy) with the writer named in argv[1] ("stream", or "whole": one
host copy of the whole state, then save_file) to argv[2]; prints the peak-RSS growth
during the save, in bytes."""

import ctypes
import sys

import torch
from safetensors.torch import save_file

from kohakumoss.training.checkpoint.stream import save_stream

try:
    from ctypes import wintypes
except (ImportError, ValueError):
    wintypes = None

COUNT, NUMEL = 16, 4 << 20


class _Counters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
        (n, ctypes.c_size_t) for n in ("peak", "ws", "a", "b", "c", "d", "e", "f")
    ]


def _win_peak() -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    query = kernel32.K32GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    c = _Counters()
    c.cb = ctypes.sizeof(_Counters)
    if not query(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError(ctypes.get_last_error(), "K32GetProcessMemoryInfo failed")
    return c.peak


def resident() -> int:
    """Current resident set size, in bytes (Linux: VmRSS; Windows: the peak)."""
    if sys.platform == "win32":
        return _win_peak()
    with open("/proc/self/status", encoding="ascii") as f:
        line = next(ln for ln in f if ln.startswith("VmRSS:"))
    return int(line.split()[1]) * 1024


def peak_since_reset() -> int:
    """Peak RSS since the last `reset_peak` (Linux: VmHWM; Windows: the peak)."""
    if sys.platform == "win32":
        return _win_peak()
    with open("/proc/self/status", encoding="ascii") as f:
        line = next(ln for ln in f if ln.startswith("VmHWM:"))
    return int(line.split()[1]) * 1024


def reset_peak() -> None:
    """Reset the peak RSS to the current RSS (Linux /proc/self/clear_refs)."""
    if sys.platform != "win32":
        with open("/proc/self/clear_refs", "w", encoding="ascii") as f:
            f.write("5")


def main() -> None:
    writer, path = sys.argv[1], sys.argv[2]
    state = {f"t{i}": torch.full((2 * NUMEL,), float(i))[::2] for i in range(COUNT)}
    reset_peak()
    base = resident()
    if writer == "stream":
        save_stream(state, path)
    else:
        save_file({k: v.contiguous() for k, v in state.items()}, path)
    print(peak_since_reset() - base)


if __name__ == "__main__":
    main()
