"""Every submodule of each kernel backend whose package is installed imports cleanly:
nothing in `backends.<backend>.FAILED`, so a broken kernel module cannot ship and run
silently on the next backend."""

import importlib
import importlib.util

import pytest

from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def check_clean(failed: dict[str, str]) -> None:
    assert not failed, f"backend submodules failed to import: {failed}"


@pytest.mark.parametrize("backend", ["triton", "tilelang"])
def test_backend_submodules_import(backend):
    if importlib.util.find_spec(backend) is None:
        pytest.skip(f"{backend} is not installed")
    module = importlib.import_module(f"kohakumoss_kernels.backends.{backend}")
    check_clean(module.FAILED)


def test_negative_failed_submodule():
    must_fail(lambda: check_clean({"norm": "ImportError: no module x"}), "FAILED")
