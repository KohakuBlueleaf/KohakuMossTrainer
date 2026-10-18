"""Comparison grids: the base model then each adapter checkpoint (the weights), each
over the same prompts with the same seeds and sampler settings, composed with the
prompts on the columns (or the rows). The DiT is loaded once; adapters are loaded and
removed per weight."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from kohakumoss import adapters as adapter_api
from kohakumoss.inference.pipeline import DEFAULT_CFG
from kohakumoss.logging.log import get_logger
from kohakumoss.model.numerics import output_row_scales

log = get_logger(__name__)

STEP_RE = re.compile(r"step[_-]?0*(\d+)")
FONT_SCALE = 32  # cell width / font size in pixels
MIN_FONT = 14
LINE = 1.25  # line height / font size
BACKGROUND = (255, 255, 255)
INK = (0, 0, 0)


@dataclass(frozen=True)
class Column:
    label: str
    adapter: tuple[str, float] | None  # (path, multiplier); None = the base model


def column_label(path: str) -> str:
    """`step N` for a training checkpoint path, else the file stem."""
    found = STEP_RE.search(Path(path).as_posix())
    return f"step {int(found.group(1))}" if found else Path(path).stem


def columns(
    checkpoints: Sequence[tuple[str, float]], labels: Sequence[str] = ()
) -> list[Column]:
    """The base column, then one column per checkpoint (label from `labels` or path)."""
    out = [Column("base", None)]
    for i, (path, mult) in enumerate(checkpoints):
        label = labels[i] if i < len(labels) else column_label(path)
        out.append(Column(label, (str(path), float(mult))))
    return out


def render_columns(
    pipe, cols: Sequence[Column], prompts: Sequence[str], mode: str = "bypass", **gen
) -> list[list[Image.Image]]:
    """[column][row] images: the prompts (and `negative_prompts`) encoded once, then
    `pipe.generate(**gen)` on those embeddings once per column, with that column's
    adapter loaded on `pipe.backbone` and removed after it."""
    negatives = gen.pop("negative_prompts", None)
    pos, neg = pipe.embeddings(list(prompts), negatives, gen.get("cfg", DEFAULT_CFG))
    gen = {**gen, "prompt_embeds": pos, "negative_embeds": neg}
    out = []
    for c, col in enumerate(cols, start=1):
        log.info(f"grid column {c}/{len(cols)}: {col.label} {col.adapter or ''}")
        stack = None
        if col.adapter is not None:
            path, mult = col.adapter
            scales = output_row_scales(pipe.backbone)
            stack = adapter_api.load(
                pipe.backbone, path, mult, mode=mode, row_scales=scales
            )
            pipe.backbone.requires_grad_(False)
            pipe.backbone.bind_ops()
        try:
            out.append(list(pipe.generate(**gen).images))
        finally:
            if stack is not None:
                stack.remove()
                pipe.backbone.bind_ops()
    return out


def truncate(text: str, chars: int) -> str:
    """ASCII `text` cut to `chars` characters (`...` marks a cut)."""
    text = text.encode("ascii", "replace").decode("ascii")
    return text if len(text) <= chars else text[: max(chars - 3, 0)] + "..."


@dataclass(frozen=True)
class Sheet:
    """Geometry of a composed grid: cell origins per grid row / column, the side label
    width, the header height, the font size and every wrapped label."""

    xs: tuple[int, ...]
    ys: tuple[int, ...]
    widths: tuple[int, ...]
    heights: tuple[int, ...]
    side: int
    header: int
    font_size: int
    col_lines: tuple[tuple[str, ...], ...]
    row_lines: tuple[tuple[str, ...], ...]

    @property
    def size(self) -> tuple[int, int]:
        return self.side + sum(self.widths), self.header + sum(self.heights)


def _fit(draw, text: str, font, width: int) -> str:
    """`text` cut (with `...`) until it is at most `width` pixels wide."""
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "...", font=font) > width:
        text = text[:-1]
    return text + "..."


def wrap(draw, text: str, font, width: int, max_lines: int) -> tuple[str, ...]:
    """ASCII `text` word-wrapped to `width` pixels, at most `max_lines` lines (the last
    one ends in `...` when text was cut)."""
    words = truncate(text, len(text)).split()
    lines, line = [], ""
    for word in words:
        joined = f"{line} {word}" if line else word
        if line and draw.textlength(joined, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = joined
    lines.append(line)
    cut = len(lines) > max_lines
    lines = [_fit(draw, x, font, width) for x in lines[:max_lines]]
    if cut:
        lines[-1] = _fit(draw, lines[-1] + " ...", font, width)
    return tuple(lines)


def _cells(images, prompt_axis: str):
    """(cells[grid row][grid col], swap): `images` is [weight][prompt]; prompt_axis
    "cols" puts weights on the rows, "rows" puts prompts on the rows."""
    match prompt_axis:
        case "cols":
            return [list(col) for col in images], False
        case "rows":
            return [list(row) for row in zip(*images)], True
        case _:
            raise ValueError(
                f"prompt_axis must be 'cols' or 'rows', got {prompt_axis!r}"
            )


def plan(
    images: Sequence[Sequence[Image.Image]],
    weight_labels: Sequence[str],
    prompt_labels: Sequence[str],
    prompt_axis: str = "cols",
    font_size: int = 0,
    max_lines: int = 8,
) -> Sheet:
    """The layout of `compose` (see there) without drawing it."""
    cells, swap = _cells(images, prompt_axis)
    col_labels, row_labels = (
        (weight_labels, prompt_labels) if swap else (prompt_labels, weight_labels)
    )
    widths = [max(row[j].width for row in cells) for j in range(len(cells[0]))]
    heights = [max(im.height for im in row) for row in cells]
    size = font_size or max(MIN_FONT, min(widths) // FONT_SCALE)
    font = ImageFont.load_default(size=size)
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    line_h = int(size * LINE)
    pad = max(4, size // 2)
    side_cap = min(widths) if swap else min(widths) // 2
    longest = max(draw.textlength(truncate(t, len(t)), font=font) for t in row_labels)
    side = int(min(max(longest + 2 * pad, size * 6), side_cap))
    col_lines = tuple(
        wrap(draw, t, font, w - 2 * pad, max_lines) for t, w in zip(col_labels, widths)
    )
    row_lines = tuple(
        wrap(draw, t, font, side - 2 * pad, max_lines) for t in row_labels
    )
    header = max(len(lines) for lines in col_lines) * line_h + 2 * pad
    heights = [
        max(h, len(lines) * line_h + 2 * pad) for h, lines in zip(heights, row_lines)
    ]
    xs = tuple(side + sum(widths[:j]) for j in range(len(widths)))
    ys = tuple(header + sum(heights[:i]) for i in range(len(heights)))
    return Sheet(
        xs, ys, tuple(widths), tuple(heights), side, header, size, col_lines, row_lines
    )


def compose(
    images: Sequence[Sequence[Image.Image]],
    weight_labels: Sequence[str],
    prompt_labels: Sequence[str],
    prompt_axis: str = "cols",
    font_size: int = 0,
    max_lines: int = 8,
) -> Image.Image:
    """One image of `images[weight][prompt]`: with `prompt_axis` "cols" the prompts head
    the columns and the weights the rows ("rows" swaps them). Labels are word-wrapped
    to their cell (at most `max_lines` lines) in a font scaled to the image width
    (`font_size` 0) or of `font_size` pixels."""
    sheet = plan(
        images, weight_labels, prompt_labels, prompt_axis, font_size, max_lines
    )
    cells, _ = _cells(images, prompt_axis)
    font = ImageFont.load_default(size=sheet.font_size)
    line_h = int(sheet.font_size * LINE)
    pad = max(4, sheet.font_size // 2)
    out = Image.new("RGB", sheet.size, BACKGROUND)
    draw = ImageDraw.Draw(out)
    for x, lines in zip(sheet.xs, sheet.col_lines):
        for k, line in enumerate(lines):
            draw.text((x + pad, pad + k * line_h), line, fill=INK, font=font)
    for i, (y, lines) in enumerate(zip(sheet.ys, sheet.row_lines)):
        for k, line in enumerate(lines):
            draw.text((pad, y + pad + k * line_h), line, fill=INK, font=font)
        for j, x in enumerate(sheet.xs):
            out.paste(cells[i][j].convert("RGB"), (x, y))
    return out
