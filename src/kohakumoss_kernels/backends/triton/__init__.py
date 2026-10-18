"""Triton backend (sm80+ CUDA, Intel XPU). Every public module of the package tree
registers its ops on import; adding an op is adding a file. A module whose import fails
on a missing module (ImportError) is recorded in `FAILED`, logged at WARNING, and the
rest still register; any other error in a module propagates. The backend's probe per
device is `_status.check` (one minimal kernel)."""

import importlib
import pkgutil

import triton  # noqa: F401  (an ImportError marks the whole backend unavailable)

from kohakumoss_common.log import get_logger
from kohakumoss_kernels import status
from kohakumoss_kernels.backends.triton import _status
from kohakumoss_kernels.hardware.toolchain import ensure_toolchain

TOOLCHAIN = ensure_toolchain()
FAILED: dict[str, str] = {}
status.register_check("triton", _status.check)

for _info in pkgutil.walk_packages(__path__, f"{__name__}."):
    _name = _info.name.removeprefix(f"{__name__}.")
    if _info.ispkg or any(p.startswith("_") for p in _name.split(".")):
        continue
    try:
        importlib.import_module(_info.name)
    except ImportError as _err:
        FAILED[_name] = f"{type(_err).__name__}: {_err}"
        get_logger(__name__).warning(
            f"kohakumoss_kernels triton.{_name}: {FAILED[_name]}"
        )
