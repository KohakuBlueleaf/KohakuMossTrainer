"""Intel Triton (triton-xpu) driver-source reads and writes in UTF-8 on any console
code page (`utf8_sources`). See docs/guides/install-windows.md#compiler-setup.
"""

import builtins
import sys
from pathlib import Path

try:
    from triton.backends.intel import compiler as _compiler
    from triton.backends.intel import driver as _driver
    from triton.backends.intel import extension_utils as _extension_utils

    MODULES = (_driver, _compiler, _extension_utils)
except ImportError:
    MODULES = ()


class Utf8Path(type(Path())):
    """`Path` whose text reads default to UTF-8."""

    def read_text(self, encoding=None, *args, **kwargs):
        return super().read_text(encoding or "utf-8", *args, **kwargs)


def utf8_open(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):
    """`open` whose text modes default to UTF-8."""
    if "b" not in mode and encoding is None:
        encoding = "utf-8"
    return builtins.open(file, mode, buffering, encoding, *args, **kwargs)


def utf8_sources() -> str:
    """Make Intel Triton's source reads / writes UTF-8 (no-op in Python UTF-8 mode or
    without Intel Triton); returns what was done."""
    if not MODULES:
        return "no intel triton"
    if sys.flags.utf8_mode:
        return "python utf-8 mode"
    for module in MODULES:
        module.Path = Utf8Path
        module.open = utf8_open
    return "intel triton sources read as utf-8"
