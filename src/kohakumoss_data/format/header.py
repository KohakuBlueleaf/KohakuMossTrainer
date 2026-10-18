"""Header-only image probing: size and alpha without decoding pixels."""

import io

from PIL import Image

ALPHA_MODES = {"RGBA", "LA", "PA", "RGBa", "La"}


def has_alpha(img: Image.Image) -> bool:
    """Whether `img` carries transparency (an alpha mode, or a palette with one)."""
    return img.mode in ALPHA_MODES or (img.mode == "P" and "transparency" in img.info)


def probe_image(source: str | bytes) -> tuple[int, int, bool]:
    """(w, h, has_alpha) after EXIF orientation, from the header only."""
    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(stream) as img:
        w, h = img.size
        orientation = img.getexif().get(0x0112, 1)
        alpha = has_alpha(img)
    if orientation in (5, 6, 7, 8):
        w, h = h, w
    return w, h, alpha
