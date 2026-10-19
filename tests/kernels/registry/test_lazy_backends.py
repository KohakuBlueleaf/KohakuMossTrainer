"""Importing kohakumoss_kernels does no backend work: no backend package, no host
toolchain setup, no compile-cache key tag until the first kernel build (`resolve`),
which does all three; an object the first build's caller held is freed by reference
counting once the caller returns (automatic gc off), although the build imported
`torch._dynamo`. Each case runs in a fresh process."""

import subprocess
import sys

import pytest

from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(120)]
CHECK = """
import sys
import torch
import kohakumoss_kernels
from kohakumoss_kernels.hardware.toolchain import ensure_toolchain
{prelude}
before = torch.compiler.config.cache_key_tag
assert "kohakumoss_kernels.backends.eager" not in sys.modules, "backends imported"
assert ensure_toolchain.cache_info().currsize == 0, "toolchain set up at import"
assert "kohakumoss-" not in before, "compile-cache tag set at import"
kohakumoss_kernels.resolve("rms_norm", "cpu", torch.float32)
assert "kohakumoss_kernels.backends.eager" in sys.modules
assert "kohakumoss-" in torch.compiler.config.cache_key_tag
print("ok")
"""


def check_lazy(prelude: str = "") -> None:
    out = subprocess.run(
        [sys.executable, "-c", CHECK.format(prelude=prelude)],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-2000:]


def test_import_is_lazy():
    check_lazy()


def test_negative_eager_backend_load():
    must_fail(lambda: check_lazy("kohakumoss_kernels.load_backends()"), "eager load")


RELEASE = """
import gc
import sys
import weakref
import torch
import kohakumoss_kernels
{prelude}


class Held:
    pass


def build():
    held = Held()
    kohakumoss_kernels.resolve("rms_norm", "cpu", torch.float32)
    return weakref.ref(held)


assert "torch._dynamo" not in sys.modules, "torch._dynamo imported before the build"
gc.disable()
ref = build()
assert "torch._dynamo" in sys.modules, "the first build did not import torch._dynamo"
assert ref() is None, "the first build's caller frame outlived it"
print("ok")
"""


def check_caller_released(prelude: str = "") -> None:
    out = subprocess.run(
        [sys.executable, "-c", RELEASE.format(prelude=prelude)],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-2000:]


NO_PASS = """
import types
import kohakumoss_kernels.registry as registry
registry.gc = types.SimpleNamespace(collect=lambda: 0)
"""


def test_first_build_releases_its_caller():
    check_caller_released()
    must_fail(lambda: check_caller_released(NO_PASS), "no gc pass after the import")
