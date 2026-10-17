"""KohakuMoss's toolchain and LyCORIS's host_env agree on nvcc's host compiler.

Whichever of `ensure_toolchain()` and `import lycoris.kernels` runs first, the compiler
TileLang hands nvcc (`-ccbin`, cached at its first query) is MSVC `cl`, with a clang-cl
discoverable. Each order runs in a fresh process: both settings are process-global.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._util.negative import must_fail

pytestmark = [
    pytest.mark.windows,
    pytest.mark.skipif(
        importlib.util.find_spec("tilelang") is None, reason="tilelang not installed"
    ),
]

TOOLCHAIN = (
    "from kohakumoss_kernels.hardware.toolchain import ensure_toolchain\n"
    "ensure_toolchain()\n"
)
LYCORIS = "import lycoris.kernels\n"
QUERY = "from tilelang.contrib.cc import get_cplus_compiler\nget_cplus_compiler()\n"
REPORT = (
    "from tilelang.contrib.cc import get_cplus_compiler\nprint(get_cplus_compiler())\n"
)
ORDERS = {
    "toolchain-then-lycoris": TOOLCHAIN + LYCORIS,
    "lycoris-then-toolchain": LYCORIS + TOOLCHAIN,
    "lycoris-compile-then-toolchain": LYCORIS + QUERY + TOOLCHAIN,
}


def _host_compiler(script: str, **env) -> str:
    """The nvcc host compiler TileLang settles on after `script`, in a fresh process
    with no CXX / CC / TILELANG_DISABLE_CLANG_CL and a clang-cl on CLANG_CL."""
    base = {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in ("CXX", "CC", "TILELANG_DISABLE_CLANG_CL")
    }
    base["CLANG_CL"] = sys.executable
    base.update(env)
    out = subprocess.run(
        [sys.executable, "-c", script + REPORT],
        capture_output=True,
        text=True,
        env=base,
        check=True,
    ).stdout
    return out.strip().splitlines()[-1]


def _check_msvc(compiler: str) -> None:
    assert Path(compiler).name.lower() == "cl.exe", f"nvcc host compiler {compiler}"


@pytest.mark.parametrize("order", sorted(ORDERS))
def test_msvc_either_order(order):
    _check_msvc(_host_compiler(ORDERS[order]))


def test_check_rejects_clang_cl():
    script = LYCORIS + QUERY
    must_fail(
        lambda: _check_msvc(_host_compiler(script, TILELANG_DISABLE_CLANG_CL="0")),
        "clang-cl cached as nvcc's host compiler",
    )
