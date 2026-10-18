"""Image decode: EXIF orientation, crop, alpha compositing, resize to the target; the
background colour and resampling filter are resolved once per `Decoder`."""

import io

import numpy as np
from PIL import Image, ImageOps

from kohakumoss_data.format.header import has_alpha

BACKGROUNDS = {"white": (255, 255, 255), "black": (0, 0, 0), "gray": (127, 127, 127)}
RESAMPLE = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "box": Image.Resampling.BOX,
}


def background_rgb(background) -> tuple[int, int, int]:
    """`white | black | gray | (r, g, b)` -> RGB tuple."""
    if isinstance(background, str):
        return BACKGROUNDS[background]
    return tuple(int(c) for c in background)


def to_rgb(img: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """RGB image; transparent pixels are composited over `background`."""
    if has_alpha(img):
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
            ImageOps.exif_transpose(raw, in_place=True)
            img = to_rgb(raw.crop(crop_ltrb), self.background)
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
