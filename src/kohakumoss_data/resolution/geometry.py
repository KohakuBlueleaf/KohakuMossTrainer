"""Target sizes and crop boxes shared by every planner: sizes are (h, w) pixels,
multiples of `step`; crop boxes are (l, t, r, b) source pixels with the target's aspect
(cropped, then resized)."""

import math
from dataclasses import dataclass

import numpy as np

# Pixels per latent token along each side: the VAE's /8 times the 2x2 patchify.
LATENT_FACTOR = 16


@dataclass(frozen=True)
class Geometry:
    """`aspect_range` bounds w/h; `upscale` is `never` or `to_area`; `crop` is
    `center` or `random`; `latent_factor` maps pixels to latent tokens."""

    area: int = 1024 * 1024
    step: int = LATENT_FACTOR
    aspect_range: tuple[float, float] = (0.25, 4.0)
    upscale: str = "never"
    crop: str = "center"
    latent_factor: int = LATENT_FACTOR

    def __post_init__(self):
        if self.step % self.latent_factor:
            raise ValueError("step must be a multiple of latent_factor")
        if self.upscale not in ("never", "to_area"):
            raise ValueError(f"upscale must be never|to_area, got {self.upscale}")
        if self.crop not in ("center", "random"):
            raise ValueError(f"crop must be center|random, got {self.crop}")

    def tokens(self, hw: tuple[int, int]) -> int:
        """Latent tokens of a (h, w) pixel size."""
        return (hw[0] // self.latent_factor) * (hw[1] // self.latent_factor)

    def clamp_aspect(self, w: int, h: int) -> tuple[int, int]:
        """Largest centred sub-size of (w, h) whose aspect lies in `aspect_range`."""
        lo, hi = self.aspect_range
        aspect = w / h
        if aspect > hi:
            return max(1, round(h * hi)), h
        if aspect < lo:
            return w, max(1, round(w / lo))
        return w, h

    def fit(self, w: int, h: int, area: int | None = None) -> tuple[int, int]:
        """Native-aspect target (h, w) with area <= `area`, sides multiples of step."""
        area = area or self.area
        cw, ch = self.clamp_aspect(w, h)
        scale = math.sqrt(area / (cw * ch))
        if self.upscale == "never":
            scale = min(scale, 1.0)
        s = self.step
        tw = max(s, int(cw * scale) // s * s)
        th = max(s, int(ch * scale) // s * s)
        want = math.log(cw / ch)
        options = [(th, tw), (th - s, tw), (th, tw - s)]
        options = [(h, w) for h, w in options if h >= s and w >= s]
        th, tw = min(options, key=lambda hw: abs(math.log(hw[1] / hw[0]) - want))
        lo, hi = self.aspect_range
        while tw / th > hi and tw > s:
            tw -= s
        while tw / th < lo and th > s:
            th -= s
        return th, tw

    def square(self) -> int:
        """Side of the square target: sqrt(area) rounded down to `step` (fixed size)."""
        return max(self.step, int(math.sqrt(self.area)) // self.step * self.step)

    def buckets(self) -> list[tuple[int, int]]:
        """(h, w) buckets of ~`area` spanning `aspect_range`, one per width step."""
        lo, hi = self.aspect_range
        out = set()
        w = self.step
        while w <= math.sqrt(self.area * hi) + self.step:
            h = int(self.area / w) // self.step * self.step
            if h >= self.step and lo <= w / h <= hi:
                out.add((h, w))
            w += self.step
        return sorted(out, key=lambda hw: hw[1] / hw[0])

    def box_size(self, w: int, h: int, target: tuple[int, int]) -> tuple[int, int]:
        """(cw, ch): the largest sub-size of (w, h) with the aspect of `target` (h, w)."""
        th, tw = target
        aspect = tw / th
        if w / h > aspect:
            return max(1, min(w, round(h * aspect))), h
        return w, max(1, min(h, round(w / aspect)))

    def center_box(
        self, w: int, h: int, target: tuple[int, int], rng: np.random.Generator
    ) -> tuple[int, int, int, int]:
        """Centred source box with the aspect of `target`; `rng` is not drawn from."""
        cw, ch = self.box_size(w, h, target)
        left, top = (w - cw) // 2, (h - ch) // 2
        return left, top, left + cw, top + ch

    def random_box(
        self, w: int, h: int, target: tuple[int, int], rng: np.random.Generator
    ) -> tuple[int, int, int, int]:
        """Seeded-position source box with the aspect of `target`."""
        cw, ch = self.box_size(w, h, target)
        left = int(rng.integers(0, w - cw + 1))
        top = int(rng.integers(0, h - ch + 1))
        return left, top, left + cw, top + ch

    def box_fn(self):
        """The box function selected by `crop`: `center_box` or `random_box`."""
        return {"center": self.center_box, "random": self.random_box}[self.crop]


def nearest_bucket(w: int, h: int, buckets: list[tuple[int, int]]) -> tuple[int, int]:
    """The (h, w) bucket whose log aspect is closest to w/h."""
    target = math.log(w / h)
    return min(buckets, key=lambda hw: abs(math.log(hw[1] / hw[0]) - target))
