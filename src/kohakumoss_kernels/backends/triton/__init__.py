"""Triton backend (sm80+ CUDA, Intel XPU). Every public submodule registers its ops on
import; adding an op is adding a file. A submodule that fails to import is recorded in
`FAILED` and the rest still register. The backend's health check per device is
`_health.check` (one minimal kernel)."""

import importlib
import pkgutil
import warnings

import triton  # noqa: F401  (an ImportError marks the whole backend unavailable)

from kohakumoss_kernels import health
from kohakumoss_kernels.backends.triton import _health
from kohakumoss_kernels.hardware.toolchain import ensure_toolchain

TOOLCHAIN = ensure_toolchain()
FAILED: dict[str, str] = {}
health.register_check("triton", _health.check)

for _info in pkgutil.iter_modules(__path__):
    if _info.name.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{_info.name}")
    except Exception as _err:  # noqa: BLE001
        FAILED[_info.name] = f"{type(_err).__name__}: {_err}"
        warnings.warn(f"kohakumoss_kernels triton.{_info.name}: {FAILED[_info.name]}")
