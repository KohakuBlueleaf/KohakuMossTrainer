"""Each shipped recipe trains: its knobs on the random tiny model with synthetic data
(the recipe's memory profile, optimiser, precision, parallel layout, EMA and exports
kept), two steps on the GPU, sampling exercised where the layout allows it. Shows the
trainer runs each recipe; not a precision test."""

from pathlib import Path

import pytest
import torch
from kohakuengine import load_config_file

from kohakumoss.training.entry import run
from kohakumoss.training.knobs import validate, with_precision
from tests.training.test_knobs import script_defaults

pytestmark = [pytest.mark.cuda, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[2] / "configs" / "moss" / "recipes"
RECIPES = sorted(ROOT.glob("recipes_*.py"))
TINY = dict(
    WEIGHTS="random",
    VARIANT="tiny",
    DATASETS=[{"source": "synthetic", "count": 16}],
    LATENT_CACHE="off",
    TEXT_CACHE="off",
    TARGET_AREA=256 * 256,
    MAX_TOKENS=2048,
    NUM_WORKERS=0,
    STEPS=2,
    SAVE_EVERY=2,
    LOG_EVERY=1,
    SCHEDULE={"mode": "cosine", "warmup": 1, "end": -1},
    SAMPLE_STEPS=2,
    SAMPLE_SIZE=(128, 128),
    SAMPLE_SKIP_LAYERS=(),
    SAMPLE_PROMPTS=["a cat"],
)


def recipe_knobs(path: Path, out: Path) -> dict:
    k = {**script_defaults(), **load_config_file(path).globals_dict}
    k.update(TINY, OUTPUT_DIR=str(out))
    k["SAMPLE_EVERY"] = 2 if k["SAMPLE_EVERY"] else 0
    if k["PARALLEL"] in ("pipeline", "threaded"):
        k["SAMPLE_EVERY"] = 0
    if k["PRECISION"] == "bf16" and not torch.cuda.is_bf16_supported():
        k["PRECISION"] = "auto"
    if k["PARALLEL"] != "none":
        k["DEVICES"] = tuple(range(len(k["DEVICES"])))
    return k


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: p.stem)
def test_recipe_trains(path, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    k = recipe_knobs(path, tmp_path)
    if len(k["DEVICES"]) > torch.cuda.device_count():
        pytest.skip(f"needs {len(k['DEVICES'])} GPUs")
    assert validate(k) == [], validate(k)
    run(k)
    name = with_precision(k)["RUN_NAME"] or "run"
    ck = tmp_path / name / "checkpoints" / "step_00000002"
    assert (ck / "trainable.safetensors").exists(), "no step-2 checkpoint"
    exports = list((tmp_path / name / "export").iterdir())
    assert len(exports) == len(k["EXPORT_FORMATS"]), exports
    if k["SAMPLE_EVERY"]:
        assert any((tmp_path / name / "samples").rglob("*.png")), "no samples"
