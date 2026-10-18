"""Factory-path resolution used by the cache scripts and the loader transform."""

import functools
import importlib

import pytest

from kohakumoss_data import DataConfig
from kohakumoss_data.scripting import build_encoder, data_config, import_object
from kohakumoss_data.sources.kohya import KohyaSource
from tests._util.encode_fns import ByteCodes
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def test_import_forms():
    assert import_object("kohakumoss_data.sources.kohya.KohyaSource") is KohyaSource
    assert import_object("kohakumoss_data.sources.kohya:KohyaSource") is KohyaSource
    method = import_object("kohakumoss_data.sources.kohya.KohyaSource.scan")
    assert method is KohyaSource.scan
    assert import_object("kohakumoss_data.sources.kohya:KohyaSource.scan") is method
    with pytest.raises(AttributeError):
        import_object("kohakumoss_data.sources.kohya.KohyaSource.nope")
    with pytest.raises(ImportError):
        import_object("no_such_pkg.thing")


def _reports_missing(resolve) -> None:
    try:
        resolve("needs_missing_dep.VALUE")
    except ModuleNotFoundError as err:
        assert err.name == "no_such_dependency_x", err
        return
    except ImportError as err:
        raise AssertionError(f"missing dependency reported as {err!r}") from err
    raise AssertionError("resolved a module whose dependency is missing")


def _any_prefix(path: str):
    """Resolution that treats every ModuleNotFoundError as an absent prefix."""
    parts = path.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:cut]))
        except ModuleNotFoundError:
            continue
        return functools.reduce(getattr, parts[cut:], module)
    raise ImportError(f"no importable module prefix in {path!r}")


def test_missing_dependency_is_reported(tmp_path, monkeypatch):
    (tmp_path / "needs_missing_dep.py").write_text("import no_such_dependency_x\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    _reports_missing(import_object)
    must_fail(lambda: _reports_missing(_any_prefix), "every missing module skipped")


def test_build_encoder():
    path = "tests._util.encode_fns.ByteCodes"
    enc = build_encoder(path, {"dim": 8})
    assert isinstance(enc, ByteCodes) and enc(["a"])[0].shape == (13, 8)
    with pytest.raises(TypeError):
        build_encoder("kohakumoss_data.seeds.seed_of", {})


def _resolved(cfg, want: str) -> None:
    assert cfg.cache_dir == want, cfg.cache_dir


def test_data_config_resolves_after_overrides(tmp_path):
    knobs = {"DATASETS": [{"source": "folder", "path": str(tmp_path)}]}
    _resolved(data_config(knobs, latent_cache="off", text_cache="off"), "")
    on = data_config(knobs, latent_cache="offline", text_cache="off")
    _resolved(on, str(tmp_path / "cache"))
    early = DataConfig.from_knobs(knobs)
    early.latent_cache = early.text_cache = "off"
    must_fail(lambda: _resolved(early, ""), "cache dir resolved before the overrides")
