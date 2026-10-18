"""The kohya source on images in the dataset root: skipped by default, read (repeat 1)
with `include_root`, and a scan that finds no image is an error naming the layout and
the knob."""

import pytest
from PIL import Image

from kohakumoss_data.sources.kohya import KohyaSource
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _tree(root, sub: bool):
    root.mkdir()
    Image.new("RGB", (16, 16)).save(root / "flat.png")
    if sub:
        (root / "3_painter").mkdir()
        Image.new("RGB", (16, 16)).save(root / "3_painter" / "a.png")
    return str(root)


def _reads_root(rows) -> None:
    assert {r.locator: r.weight for r in rows}.get("flat.png") == 1.0


def test_root_images(tmp_path):
    path = _tree(tmp_path / "both", True)
    rows = list(KohyaSource(path, include_root=True).scan())
    _reads_root(rows)
    assert {r.locator for r in rows} == {"flat.png", "3_painter/a.png"}
    default = list(KohyaSource(path).scan())
    assert [r.locator for r in default] == ["3_painter/a.png"]
    must_fail(lambda: _reads_root(default), "root images skipped by default")


def test_no_image_found_is_an_error(tmp_path):
    flat = _tree(tmp_path / "flat", False)
    with pytest.raises(ValueError, match="include_root=True"):
        KohyaSource(flat).walk()
    assert len(KohyaSource(flat, include_root=True).walk()) == 1
