"""Labelled image strips of the model-card figures: each image resized to a PANEL square
with a white label box and dark text top left, side by side on white with GAP gutters,
saved as RGB webp."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PANEL = 512  # panel side in pixels
GAP = 8  # gutter and margin in pixels
INK = "#333333"  # text and axes on the white background
PAPER = "#ffffff"  # figure background, opaque


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def label(img: Image.Image, text: str) -> Image.Image:
    """`img` as an RGB PANEL square with `text` in a white box, top left."""
    img = img.convert("RGB").resize((PANEL, PANEL), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    f = font(20)
    w = d.textlength(text, font=f)
    d.rectangle((0, 0, w + 16, 32), fill=PAPER)
    d.text((8, 5), text, fill=INK, font=f)
    return img


def strip(panels: list[tuple[Path, str]], dest: Path) -> bool:
    """(image, label) panels in a row as webp at `dest`; False when an image is missing."""
    if not all(p.is_file() for p, _ in panels):
        return False
    imgs = [label(Image.open(p), t) for p, t in panels]
    n = len(imgs)
    out = Image.new("RGB", (n * PANEL + (n + 1) * GAP, PANEL + 2 * GAP), PAPER)
    for i, im in enumerate(imgs):
        out.paste(im, (GAP + i * (PANEL + GAP), GAP))
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest, "WEBP", quality=88, method=6)
    return True
