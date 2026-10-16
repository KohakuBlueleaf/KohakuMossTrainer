"""Every shipped recipe composes onto the train script and validates (paths aside)."""

from pathlib import Path

import pytest
import torch
from kohakuengine import load_config_file

from kohakumoss.adapters.spec import spec_from_knobs
from kohakumoss.training.knobs import validate
from kohakumoss.training.memory.profiles import resolve
from tests.training.test_knobs import script_defaults

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2] / "configs" / "moss"
RECIPES = sorted((ROOT / "recipes").glob("recipes_*.py")) + [
    ROOT / "smoke" / "smoke_tiny.py"
]


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: p.stem)
def test_recipe_validates(path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    k = {**script_defaults(), **load_config_file(path).globals_dict}
    problems = [p for p in validate(k) if "does not exist" not in p]
    assert problems == [], problems
    assert spec_from_knobs(k).validate() == [] or k["ALGO"] == "full"
    resolve(
        k["MEMORY_PROFILE"],
        k["MEMORY_OVERRIDES"],
        device="cpu",
        trainable_base=k["ALGO"] == "full",
    )
