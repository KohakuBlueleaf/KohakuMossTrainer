"""TileLang backend (the sm70 / sm75 tensor-core path; also raced on sm80+). Every
public submodule registers its ops on import; a failing submodule is recorded in
`FAILED` and the rest still register. Without a host compiler for nvcc the backend is
marked unavailable."""

import importlib
import pkgutil
import warnings

import tilelang  # noqa: F401  (an ImportError marks the whole backend unavailable)
import torch

from kohakumoss_kernels import health
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
health.register_check("tilelang", check)

for _info in pkgutil.iter_modules(__path__):
    if _info.name.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{_info.name}")
    except Exception as _err:  # noqa: BLE001
        FAILED[_info.name] = f"{type(_err).__name__}: {_err}"
        warnings.warn(f"kohakumoss_kernels tilelang.{_info.name}: {FAILED[_info.name]}")
