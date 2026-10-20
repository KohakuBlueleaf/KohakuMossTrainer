"""CUDA runtime calls of host registration: register / unregister through torch's own
cudart binding, and the reset of the runtime's last error after a refused call
through the libcudart torch loaded (torch's binding has no cudaGetLastError). See
docs/internals/offload.md#host-arena."""

import ctypes
import sys
from functools import cache
from pathlib import Path

import torch

MAPS = Path("/proc/self/maps")


def register(ptr: int, nbytes: int) -> str | None:
    """Page-lock host range [ptr, ptr + nbytes); None, else the runtime's error text."""
    cudart = torch.cuda.cudart()
    err = cudart.cudaHostRegister(ptr, nbytes, 0)
    if err == cudart.cudaError.success:
        return None
    return f"{err.name}: {cudart.cudaGetErrorString(err)}"


def unregister(ptr: int) -> None:
    torch.cuda.cudart().cudaHostUnregister(ptr)


def mapped_libraries(maps: str, name: str) -> list[str]:
    """Paths of the mapped libraries in /proc/<pid>/maps text whose file name starts
    with `name`."""
    out = []
    for line in maps.splitlines():
        parts = line.split(maxsplit=5)
        path = parts[5].strip() if len(parts) == 6 else ""
        if path and Path(path).name.startswith(name) and path not in out:
            out.append(path)
    return out


def windows_candidates() -> list[Path]:
    """cudart DLLs next to torch's own libraries and in the NVIDIA runtime wheels."""
    torch_lib = Path(torch.__file__).parent / "lib"
    nvidia = Path(torch.__file__).parent.parent / "nvidia"
    return sorted(torch_lib.glob("cudart64_*.dll")) + sorted(
        nvidia.glob("*/bin/cudart64_*.dll")
    )


@cache
def runtime_library() -> ctypes.CDLL | None:
    """The libcudart this process has loaded (the instance whose last error torch
    reads), or None when none is found."""
    match sys.platform:
        case "linux":
            try:
                paths = mapped_libraries(MAPS.read_text(), "libcudart.so")
            except OSError:
                paths = []
            return ctypes.CDLL(paths[0]) if paths else None
        case "win32":
            kernel32 = ctypes.windll.kernel32
            for dll in windows_candidates():
                if kernel32.GetModuleHandleW(str(dll)):
                    return ctypes.CDLL(str(dll))
    return None


def reset_last_error() -> None:
    """Clear the runtime's last error a refused registration set (the next kernel
    launch check would raise it); raises when the runtime library is not found."""
    lib = runtime_library()
    if lib is None:
        raise RuntimeError(
            "cudaHostRegister was refused and this torch build has no shared CUDA "
            "runtime loaded (cudart linked statically), so the runtime's last error "
            "cannot be cleared and the pinned-chunk fallback cannot run; set "
            "PINNED=False (MEMORY_PROFILE='manual') or TEXT_ENCODE={'pinned': False}"
        )
    lib.cudaGetLastError()
