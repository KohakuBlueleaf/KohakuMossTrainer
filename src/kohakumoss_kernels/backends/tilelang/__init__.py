"""TileLang backend (the sm70 / sm75 tensor-core path; also raced on sm80+). Every
public submodule registers its ops on import; a failing submodule is recorded in
`FAILED` and the rest still register."""

import importlib
import pkgutil
import warnings

import tilelang  # noqa: F401  (an ImportError marks the whole backend unavailable)

from kohakumoss_kernels.hardware.toolchain import ensure_host_compiler

FAILED: dict[str, str] = {}
HOST = ensure_host_compiler()

for _info in pkgutil.iter_modules(__path__):
    if _info.name.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{_info.name}")
    except Exception as _err:  # noqa: BLE001
        FAILED[_info.name] = f"{type(_err).__name__}: {_err}"
        warnings.warn(f"kohakumoss_kernels tilelang.{_info.name}: {FAILED[_info.name]}")
