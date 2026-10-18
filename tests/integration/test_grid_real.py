"""The comparison-grid script on the real weights: one prompt, the base model and one
adapter, prompts on the columns. It writes the sheet, every single image and the
metadata; the adapter row differs from the base row."""

import json
from pathlib import Path

import numpy as np
import pytest
from kohakuengine import run
from PIL import Image

from tests._util.negative import must_fail

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sample" / "compare_grid.py"


def _differ(a: Path, b: Path) -> None:
    x = np.asarray(Image.open(a), dtype=np.int16)
    y = np.asarray(Image.open(b), dtype=np.int16)
    assert np.abs(x - y).mean() > 1.0, (a.name, b.name)


def test_grid_script(weights_root, adapter, tmp_path):
    out = tmp_path / "grid"
    sets = {
        "WEIGHTS": str(weights_root / "base"),
        "OUT": str(out),
        "CHECKPOINTS": adapter,
        "PROMPTS": ["a lighthouse on a rocky coast at sunset"],
        "SEEDS": "3",
        "SIZES": "512x512",
        "STEPS": "8",
        "DEVICE": "cuda",
        "TEXT_ENCODE": {"memory": "stream", "free_after_use": True},
        "MEMORY_PROFILE": "balanced",
        "GRID_LAYOUT": "prompt-cols",
    }
    run(str(SCRIPT), set_overrides=sets)
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert [c["label"] for c in meta["columns"]][0] == "base"
    assert len(meta["columns"]) == 2 and meta["rows"][0]["seed"] == 3
    names = sorted(p.name for p in out.glob("r*_c*.png"))
    assert names == ["r00_c00.png", "r00_c01.png"]
    sheet = Image.open(out / "grid.png")
    assert sheet.width >= 512 and sheet.height >= 2 * 512
    _differ(out / "r00_c00.png", out / "r00_c01.png")
    must_fail(lambda: _differ(out / "r00_c00.png", out / "r00_c00.png"), "same image")
