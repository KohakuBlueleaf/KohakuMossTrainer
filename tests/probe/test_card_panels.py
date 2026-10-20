"""Model-card image strips: RGB on white, PANEL squares with GAP gutters in the given
order, a white label box with dark text, and no strip when an image is missing."""

import pytest
from PIL import Image

from kohakumoss.probe import card_panels
from kohakumoss.probe.card_panels import GAP, PANEL, strip
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

COLORS = ((200, 30, 30), (30, 200, 30), (30, 30, 200), (90, 90, 90))


def _images(tmp_path) -> list:
    out = []
    for i, c in enumerate(COLORS):
        p = tmp_path / f"in{i}.png"
        Image.new("RGBA", (300, 200), c + (128,)).save(p)
        out.append(p)
    return out


def _check(dest) -> None:
    im = Image.open(dest)
    assert im.mode == "RGB"
    assert im.size == (4 * PANEL + 5 * GAP, PANEL + 2 * GAP)
    px = im.load()
    assert all(v > 245 for v in px[GAP // 2, GAP // 2])
    for i, c in enumerate(COLORS):
        x = GAP + i * (PANEL + GAP)
        got = px[x + PANEL // 2, GAP + PANEL // 2]
        assert all(abs(g - w) < 12 for g, w in zip(got, c)), (i, got, c)
        assert all(v > 245 for v in px[x + 2, GAP + 2])
        gutter = (
            px[x + PANEL + GAP // 2, GAP + PANEL // 2]
            if i < 3
            else px[x - GAP // 2, GAP + PANEL // 2]
        )
        assert all(v > 245 for v in gutter)


def test_strip_layout(tmp_path):
    imgs = _images(tmp_path)
    dest = tmp_path / "out" / "row.webp"
    assert strip([(p, f"col {i}") for i, p in enumerate(imgs)], dest)
    _check(dest)


def test_strip_missing_image(tmp_path):
    imgs = _images(tmp_path)
    dest = tmp_path / "row.webp"
    assert not strip([(imgs[0], "a"), (tmp_path / "none.png", "b")], dest)
    assert not dest.exists()


def test_strip_negative_dark_paper(tmp_path, monkeypatch):
    monkeypatch.setattr(card_panels, "PAPER", "#000000")
    imgs = _images(tmp_path)
    dest = tmp_path / "row.webp"
    card_panels.strip([(p, "x") for p in imgs], dest)
    must_fail(lambda: _check(dest), "black background and label box")
