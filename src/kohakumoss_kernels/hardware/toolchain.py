"""Toolchain discovery: kernel packages, CUDA, and the host C++ compiler.

On Windows nvcc accepts only MSVC; `ensure_host_compiler` points `CXX` at the newest
`cl.exe` (vswhere, then the standard install roots) so TileLang never picks up a
clang-cl from PATH and no vcvars shell is needed.
"""

import glob
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

VSWHERE = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
VS_ROOTS = (
    r"C:\Program Files\Microsoft Visual Studio",
    r"C:\Program Files (x86)\Microsoft Visual Studio",
)
CL_GLOB = r"*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"


@dataclass(frozen=True)
class Status:
    """Whether a component is usable, and why not when it is not."""

    ok: bool
    detail: str = ""


def installed(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_key(path: str) -> tuple:
    parts = Path(path).parts
    idx = next((i for i, p in enumerate(parts) if p.upper() == "MSVC"), None)
    ver = parts[idx + 1] if idx is not None and idx + 1 < len(parts) else "0"
    return tuple(int(x) if x.isdigit() else 0 for x in ver.split("."))


@cache
def msvc_compiler() -> str | None:
    """Path of the newest x64 MSVC `cl.exe`, or None."""
    found: list[str] = []
    if VSWHERE.exists():
        roots = subprocess.run(
            [str(VSWHERE), "-all", "-products", "*", "-property", "installationPath"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        for root in roots:
            found += glob.glob(
                os.path.join(root, "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64")
                + os.sep
                + "cl.exe"
            )
    for root in VS_ROOTS:
        found += glob.glob(os.path.join(root, CL_GLOB))
    return max(found, key=_version_key) if found else None


def ensure_host_compiler() -> Status:
    """Make nvcc's host compiler resolvable; sets `CXX` on Windows when unset."""
    if os.environ.get("CXX"):
        return Status(True, f"CXX={os.environ['CXX']}")
    if sys.platform == "win32":
        compiler = msvc_compiler()
        if compiler is None:
            return Status(False, "MSVC cl.exe not found; install VS Build Tools (C++)")
        os.environ["CXX"] = compiler
        return Status(True, f"CXX={compiler}")
    found = shutil.which("g++") or shutil.which("clang++")
    return Status(found is not None, found or "no g++ / clang++ on PATH")


@cache
def nvcc_version() -> str | None:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = shutil.which("nvcc") or (str(Path(home, "bin", "nvcc")) if home else None)
    if nvcc is None:
        return None
    try:
        out = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return None
    line = next((ln for ln in out.splitlines() if "release" in ln), "")
    return line.split("release")[-1].strip() or None


def triton_status() -> Status:
    if not installed("triton"):
        hint = "triton-windows" if sys.platform == "win32" else "triton"
        return Status(False, f"triton not installed (pip install {hint})")
    return Status(True, version("triton") or version("triton-windows") or "?")


def tilelang_status() -> Status:
    if not installed("tilelang"):
        return Status(False, "tilelang not installed")
    host = ensure_host_compiler()
    if not host.ok:
        return host
    return Status(True, version("tilelang") or "?")


def report() -> dict[str, str]:
    """Versions and component status, one line each (the doctor's view)."""
    out = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "torch": torch.__version__,
        "torch.cuda": str(torch.version.cuda),
        "nvcc": str(nvcc_version()),
        "cuda.available": str(torch.cuda.is_available()),
        "xpu.available": str(hasattr(torch, "xpu") and torch.xpu.is_available()),
    }
    for name, status in (("triton", triton_status()), ("tilelang", tilelang_status())):
        out[name] = ("ok " if status.ok else "missing: ") + status.detail
    if sys.platform == "win32":
        out["msvc"] = str(msvc_compiler())
    return out
