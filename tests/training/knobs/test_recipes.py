"""Every shipped recipe composes onto the train script, validates (paths aside), and
builds its data side: the cache location resolves and a dataset with the recipe's data
knobs plans its first batch."""

import sys
from pathlib import Path

import pytest
import torch
from kohakuengine import load_config_file

from kohakumoss.adapters.build.config import config_from_knobs
from kohakumoss.memory.profiles import resolve
from kohakumoss.training.knobs import validate
from kohakumoss_data import DataConfig, MossDataset, resolved_cache_dir
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import script_defaults

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[3] / "configs" / "moss"
RECIPES = sorted((ROOT / "recipes").glob("recipes_*.py"))


def _knobs(path) -> dict:
    return {**script_defaults(), **load_config_file(path).globals_dict}


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: p.stem)
def test_recipe_validates(path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "platform", "linux")  # multi-GPU recipes are Linux-only
    k = _knobs(path)
    problems = [p for p in validate(k) if "does not exist" not in p]
    assert problems == [], problems
    assert config_from_knobs(k).validate() == [] or k["ALGO"] == "full"
    resolve(
        k["MEMORY_PROFILE"],
        k["MEMORY_OVERRIDES"],
        device="cpu",
        trainable_base=k["ALGO"] == "full",
    )


def _builds_data(k: dict, cache: Path) -> None:
    """The recipe's data knobs on a synthetic stand-in dataset under `cache`."""
    cfg = DataConfig.from_knobs(k)
    caches_on = "off" not in (cfg.latent_cache, cfg.text_cache) or (
        cfg.latent_cache != cfg.text_cache
    )
    assert not caches_on or resolved_cache_dir(cfg), "no cache location"
    stand_in = {
        **k,
        "DATASETS": [{"source": "synthetic", "count": 8}],
        "CACHE_DIR": str(cache),
        "NUM_WORKERS": 0,
    }
    ds = MossDataset(DataConfig.from_knobs(stand_in))
    assert ds.planner.plan(0, 0, 1).items


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: p.stem)
def test_recipe_data_builds(path, tmp_path):
    k = _knobs(path)
    _builds_data(k, tmp_path / "cache")
    if k["LATENT_CACHE"] != "off":
        unresolved = {**k, "DATASETS": [{"source": "synthetic"}], "CACHE_DIR": ""}
        must_fail(lambda: _builds_data(unresolved, tmp_path / "c2"), "no cache dir")
