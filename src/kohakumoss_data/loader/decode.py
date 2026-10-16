"""Image decode: EXIF orientation, alpha compositing, crop then resize to the target.

`Decoder` resolves the background colour and the resampling filter once; per image
only the image's own properties (mode, size, the plan's flip) decide what runs.
"""

import io

import numpy as np
from PIL import Image, ImageOps

BACKGROUNDS = {"white": (255, 255, 255), "black": (0, 0, 0), "gray": (127, 127, 127)}
RESAMPLE = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "box": Image.Resampling.BOX,
}
ALPHA_MODES = ("RGBA", "LA", "PA")


def background_rgb(spec) -> tuple[int, int, int]:
    """`white | black | gray | (r, g, b)` -> RGB tuple."""
    if isinstance(spec, str):
        return BACKGROUNDS[spec]
    return tuple(int(c) for c in spec)


def to_rgb(img: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """RGB image; transparent pixels are composited over `background`."""
    if img.mode in ALPHA_MODES or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        base = Image.new("RGBA", rgba.size, background + (255,))
        return Image.alpha_composite(base, rgba).convert("RGB")
    return img.convert("RGB")


class Decoder:
    """bytes -> uint8 [H, W, 3] of the crop box resized to `target_hw` (h, w)."""

    def __init__(self, background="white", resample: str = "lanczos"):
        self.background = background_rgb(background)
        self.resample = RESAMPLE[resample]

    def __call__(
        self,
        data: bytes,
        crop_ltrb: tuple[int, int, int, int],
        target_hw: tuple[int, int],
        flip: bool,
    ) -> np.ndarray:
        with Image.open(io.BytesIO(data)) as raw:
            img = to_rgb(ImageOps.exif_transpose(raw), self.background)
        img = img.crop(crop_ltrb)
        th, tw = target_hw
        if img.size != (tw, th):
            img = img.resize((tw, th), self.resample, reducing_gap=3.0)
        if flip:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return np.array(img, dtype=np.uint8)


def decode_image(
    data: bytes,
    crop_ltrb: tuple[int, int, int, int],
    target_hw: tuple[int, int],
    flip: bool,
    background=(255, 255, 255),
    resample: str = "lanczos",
) -> np.ndarray:
    """One-off decode (builds a Decoder; datasets hold a bound one instead)."""
    return Decoder(background, resample)(data, crop_ltrb, target_hw, flip)
