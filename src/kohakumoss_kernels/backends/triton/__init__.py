"""Triton backend (sm80+ CUDA, Intel XPU). Every public submodule registers its ops on
import; adding an op is adding a file. A submodule that fails to import is recorded in
`FAILED` and the rest still register."""

import importlib
import pkgutil
import warnings

import triton  # noqa: F401  (an ImportError marks the whole backend unavailable)

FAILED: dict[str, str] = {}

for _info in pkgutil.iter_modules(__path__):
    if _info.name.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{_info.name}")
    except Exception as _err:  # noqa: BLE001
        FAILED[_info.name] = f"{type(_err).__name__}: {_err}"
        warnings.warn(f"kohakumoss_kernels triton.{_info.name}: {FAILED[_info.name]}")
