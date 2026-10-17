"""Importing kohakumoss.adapters against a LyCORIS without the tier API / half-matmul
policy fails with the required version named."""

import importlib
import sys

import lycoris.kernels as lyk
import pytest

from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _reimport(monkeypatch):
    for name in [n for n in sys.modules if n.startswith("kohakumoss.adapters")]:
        monkeypatch.delitem(sys.modules, name)
    return importlib.import_module("kohakumoss.adapters")


def _imports_cleanly(monkeypatch):
    try:
        _reimport(monkeypatch)
    except ImportError as e:
        raise AssertionError(e) from e


def test_the_current_lycoris_imports(monkeypatch):
    _imports_cleanly(monkeypatch)


def test_an_old_lycoris_is_named(monkeypatch):
    monkeypatch.delattr(lyk, "set_half_matmul")
    with pytest.raises(ImportError, match=r"lycoris-lora >= 4\.1\.0\.dev0"):
        _reimport(monkeypatch)
    must_fail(
        lambda: _imports_cleanly(monkeypatch), "a LyCORIS without set_half_matmul"
    )
