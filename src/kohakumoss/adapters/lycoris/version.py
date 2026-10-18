"""Installed-LyCORIS check: lycoris-lora >= LYCORIS_MIN with the public kernel API
(`lycoris.kernels.autograd`, `dispatch`, `limits`, `select`). Imported first by
`kohakumoss.adapters`, so an older LyCORIS fails with the version named."""

import importlib
from importlib.metadata import PackageNotFoundError, version

from packaging.version import InvalidVersion, Version

LYCORIS_MIN = "4.1.2"
NOT_FOUND = "no installed distribution"
REQUIRED = (
    "lycoris.kernels.autograd",
    "lycoris.kernels.dispatch",
    "lycoris.kernels.limits",
    "lycoris.kernels.select",
)


def lycoris_version() -> str:
    try:
        return version("lycoris-lora")
    except PackageNotFoundError:
        return NOT_FOUND


def older(found: str) -> bool:
    """Whether installed version `found` is below LYCORIS_MIN (an unparsable one is)."""
    try:
        return Version(found) < Version(LYCORIS_MIN)
    except InvalidVersion:
        return True


def check() -> None:
    """Raise ImportError naming the required version when a module is missing or the
    installed version is below LYCORIS_MIN."""
    found = lycoris_version()
    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except ImportError as e:
            raise ImportError(
                f"kohakumoss needs lycoris-lora >= {LYCORIS_MIN} ({name}); "
                f"found {found}"
            ) from e
    if found != NOT_FOUND and older(found):
        raise ImportError(
            f"kohakumoss needs lycoris-lora >= {LYCORIS_MIN}; found {found}"
        )


check()
