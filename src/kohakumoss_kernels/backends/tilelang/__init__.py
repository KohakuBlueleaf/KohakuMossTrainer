"""TileLang backend (the sm70 / sm75 tensor-core path; next after Triton on sm80+).
Every public module of the package tree registers its ops on import; a module whose
import fails on a missing module (ImportError) is recorded in `FAILED`, logged at
WARNING, and the rest still register; any other error in a module propagates. Without a
host compiler for nvcc the backend is marked unavailable."""

import importlib
import pkgutil

import tilelang  # noqa: F401  (an ImportError marks the whole backend unavailable)
import torch

from kohakumoss_common.log import get_logger
from kohakumoss_kernels import status
from kohakumoss_kernels.hardware.toolchain import ensure_toolchain
from kohakumoss_kernels.registry import mark_unavailable

FAILED: dict[str, str] = {}
TOOLCHAIN = ensure_toolchain()


def check(device: torch.device) -> None:
    """Raise unless `device` is CUDA and nvcc's host compiler is set up."""
    if device.type != "cuda":
        raise RuntimeError(f"tilelang does not run on {device.type}")
    if not TOOLCHAIN.ok:
        raise RuntimeError(TOOLCHAIN.detail)


if not TOOLCHAIN.ok:
    mark_unavailable("tilelang", TOOLCHAIN.detail)
status.register_check("tilelang", check)

for _info in pkgutil.walk_packages(__path__, f"{__name__}."):
    _name = _info.name.removeprefix(f"{__name__}.")
    if _info.ispkg or any(p.startswith("_") for p in _name.split(".")):
        continue
    try:
        importlib.import_module(_info.name)
    except ImportError as _err:
        FAILED[_name] = f"{type(_err).__name__}: {_err}"
        get_logger(__name__).warning(
            f"kohakumoss_kernels tilelang.{_name}: {FAILED[_name]}"
        )
