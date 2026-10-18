"""Importing kohakumoss.adapters against a LyCORIS without the public kernel API fails
with the required version named, and that version is pyproject.toml's pin."""

import importlib
import importlib.metadata
import re
import sys
from pathlib import Path

import pytest

from kohakumoss.adapters.lycoris import version as lycoris_version
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


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


@pytest.mark.parametrize("name", lycoris_version.REQUIRED)
def test_an_old_lycoris_is_named(monkeypatch, name):
    monkeypatch.setitem(sys.modules, name, None)
    pinned = re.escape(f"lycoris-lora >= {lycoris_version.LYCORIS_MIN}")
    with pytest.raises(ImportError, match=pinned):
        _reimport(monkeypatch)
    must_fail(lambda: _imports_cleanly(monkeypatch), f"a LyCORIS without {name}")


def _old_installed(monkeypatch) -> None:
    """lycoris-lora 4.1.1 installed, for a fresh import and for this module's check."""
    real = importlib.metadata.version

    def fake(name):
        return "4.1.1" if name == "lycoris-lora" else real(name)

    monkeypatch.setattr(importlib.metadata, "version", fake)
    monkeypatch.setattr(lycoris_version, "version", fake)


def check_old_version_named(check) -> None:
    pinned = f"lycoris-lora >= {lycoris_version.LYCORIS_MIN}; found 4.1.1"
    try:
        check()
    except ImportError as err:
        assert pinned in str(err), err
        return
    raise AssertionError("an old LyCORIS imported without the named version")


def test_an_old_version_is_named(monkeypatch):
    _old_installed(monkeypatch)
    check_old_version_named(lambda: _reimport(monkeypatch))
    check_old_version_named(lycoris_version.check)
    must_fail(lambda: check_old_version_named(lambda: None), "the check skipped")


def _pinned(minimum: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    pins = re.findall(r'"lycoris-lora>=([^",;\s]+)', text)
    assert pins == [minimum], pins


def test_the_named_version_is_the_pin():
    _pinned(lycoris_version.LYCORIS_MIN)
    must_fail(lambda: _pinned("4.1.0.dev0"), "a stale minimum")
