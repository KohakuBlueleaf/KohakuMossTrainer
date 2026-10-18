"""The train starter's sample prompt: `--prompt` when given, else the first caption of
the dataset (only its first rows read), else the base prompt with a warning."""

from pathlib import Path

import pytest
from PIL import Image

from kohakumoss.cli.config import starter
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]


def kohya_dataset(root: Path, tags: str) -> Path:
    folder = root / "data" / "5_painter"
    folder.mkdir(parents=True)
    Image.new("RGB", (32, 32), (90, 30, 10)).save(folder / "a.png")
    (folder / "a.txt").write_text(tags, encoding="utf-8")
    return root / "data"


def prompts_of(data: Path, prompts=()) -> list[str]:
    choice = starter.TrainChoice(
        weights=str(data.parent / "w"), data=str(data), vram=24, prompts=prompts
    )
    return starter.train_starter(ROOT, choice).own.get("SAMPLE_PROMPTS", [])


def _from_the_data(prompts: list[str]) -> None:
    """One prompt holding the folder's artist (kept by the kohya caption layer)."""
    assert len(prompts) == 1 and "painter" in prompts[0], prompts


def test_the_sample_prompt_comes_from_the_dataset(tmp_path):
    data = kohya_dataset(tmp_path, "lighthouse, dusk, sea")
    _from_the_data(prompts_of(data))
    assert prompts_of(data, ("a given prompt",)) == ["a given prompt"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert prompts_of(empty) == []
    must_fail(lambda: _from_the_data(prompts_of(empty)), "no caption to read")
