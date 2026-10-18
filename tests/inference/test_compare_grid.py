"""Comparison-grid layout, model-free: weights / prompts on the chosen axes, wrapped
labels inside their cells, the font scaled to the cells, the column labels, one text
encode for every column. Rendering a grid on the real model runs in
tests/integration/test_grid_real.py."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from kohakumoss.inference import grid
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def tiles(n_weights: int, n_prompts: int) -> list[list[Image.Image]]:
    """[weight][prompt] images of distinct solid colours, 64 x 96 (H x W)."""
    return [
        [
            Image.new("RGB", (96, 64), (40 * w + 10, 30 * p + 20, 7))
            for p in range(n_prompts)
        ]
        for w in range(n_weights)
    ]


def cell(sheet_img, layout, i: int, j: int) -> np.ndarray:
    x, y = layout.xs[j], layout.ys[i]
    return np.asarray(sheet_img)[y : y + 64, x : x + 96]


def placed(images, sheet_img, layout, prompt_axis: str) -> None:
    for w, col in enumerate(images):
        for p, im in enumerate(col):
            i, j = (w, p) if prompt_axis == "cols" else (p, w)
            assert np.array_equal(cell(sheet_img, layout, i, j), np.asarray(im))


@pytest.mark.parametrize("axis", ["cols", "rows"])
def test_compose_layout(axis):
    images = tiles(3, 2)
    weights = ["base", "lokr x0.5", "lokr x1"]
    prompts = [
        "0: " + "painter, a lighthouse, rocky coast, sunset, " * 6,
        "1: a dog on grass",
    ]
    layout = grid.plan(images, weights, prompts, prompt_axis=axis)
    sheet = grid.compose(images, weights, prompts, prompt_axis=axis)
    assert sheet.size == layout.size
    assert len(layout.xs) == (2 if axis == "cols" else 3)
    placed(images, sheet, layout, axis)
    other = "rows" if axis == "cols" else "cols"
    must_fail(lambda: placed(images, sheet, layout, other), "weights / prompts swapped")
    labels = layout.col_lines if axis == "cols" else layout.row_lines
    assert len(labels[0]) > 1, "a long prompt wraps over several lines"
    font = grid.ImageFont.load_default(size=layout.font_size)
    draw = grid.ImageDraw.Draw(Image.new("RGB", (1, 1)))
    limit = 96 if axis == "cols" else layout.side
    assert all(draw.textlength(x, font=font) <= limit for x in labels[0])
    assert grid.plan(
        images, weights, prompts, prompt_axis=axis, max_lines=2
    ).header <= (layout.header)
    with pytest.raises(ValueError):
        grid.plan(images, weights, prompts, prompt_axis="diagonal")


def test_font_scales_with_cells():
    small = grid.plan(tiles(1, 1), ["base"], ["p"])
    big_tiles = [[Image.new("RGB", (1024, 1024))]]
    big = grid.plan(big_tiles, ["base"], ["p"])
    assert small.font_size == grid.MIN_FONT
    assert big.font_size == 1024 // grid.FONT_SCALE
    assert grid.plan(big_tiles, ["base"], ["p"], font_size=20).font_size == 20


class Pipe:
    """The calls a grid makes on a pipeline: encodings counted, each generate's
    arguments kept (no model)."""

    def __init__(self):
        self.encoded, self.calls = [], []

    def embeddings(self, prompts, negatives=None, cfg=3.0):
        self.encoded.append((list(prompts), negatives, cfg))
        return [f"e:{p}" for p in prompts], [f"n:{p}" for p in negatives or prompts]

    def generate(self, prompts=None, **gen):
        self.calls.append({"prompts": prompts, **gen})
        n = len(gen.get("prompt_embeds") or prompts)
        return SimpleNamespace(images=[Image.new("RGB", (8, 8))] * n)


def _one_encode_for_every_column(render) -> None:
    pipe = Pipe()
    cols = [grid.Column("base", None), grid.Column("again", None)]
    images = render(pipe, cols, ["a", "b"], negative_prompts=["", ""], cfg=2.0)
    assert len(images) == 2 and len(images[0]) == 2
    assert pipe.encoded == [(["a", "b"], ["", ""], 2.0)], pipe.encoded
    for call in pipe.calls:
        assert call["prompts"] is None and call["prompt_embeds"] == ["e:a", "e:b"]
        assert call["negative_embeds"] == ["n:", "n:"] and call["cfg"] == 2.0


def _encode_per_column(pipe, cols, prompts, mode="bypass", **gen):
    return [list(pipe.generate(list(prompts), **gen).images) for _ in cols]


def test_columns_share_one_text_encode():
    _one_encode_for_every_column(grid.render_columns)
    must_fail(
        lambda: _one_encode_for_every_column(_encode_per_column),
        "every column encodes its prompts again",
    )


def test_column_labels():
    assert grid.column_label(
        "runs/q/checkpoints/step_00000400/trainable.safetensors"
    ) == ("step 400")
    assert grid.column_label("my_lora.safetensors") == "my_lora"
    assert grid.truncate("x" * 50, 10) == "xxxxxxx..."
