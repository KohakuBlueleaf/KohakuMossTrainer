"""Synthetic source: seeded random images and tag captions (smoke runs, tests)."""

import io
from collections.abc import Iterator

import numpy as np
from PIL import Image

from kohakumoss_data.registry import SOURCES
from kohakumoss_data.seeds import rng_of, text_hash
from kohakumoss_data.types import CatalogRow

VOCAB = tuple(f"tag{i:03d}" for i in range(256))
SIZES = ((512, 512), (640, 384), (384, 640), (768, 512), (512, 768), (1024, 1024))


@SOURCES.register("synthetic")
class SyntheticSource:
    """`count` images of seeded sizes from `sizes`; captions are 4-24 vocabulary tags."""

    def __init__(
        self,
        count: int = 64,
        seed: int = 0,
        sizes: tuple[tuple[int, int], ...] = SIZES,
        alpha_every: int = 0,
    ):
        self.count = count
        self.seed = seed
        self.sizes = tuple(tuple(s) for s in sizes)
        self.alpha_every = alpha_every

    def _size(self, i: int) -> tuple[int, int]:
        rng = rng_of("synthetic-size", self.seed, i)
        return self.sizes[int(rng.integers(len(self.sizes)))]

    def _alpha(self, i: int) -> bool:
        return bool(self.alpha_every) and i % self.alpha_every == 0

    def scan(self) -> Iterator[CatalogRow]:
        for i in range(self.count):
            w, h = self._size(i)
            rng = rng_of("synthetic-tags", self.seed, i)
            n = int(rng.integers(4, 25))
            tags = [VOCAB[j] for j in rng.choice(len(VOCAB), size=n, replace=False)]
            row = CatalogRow(
                id=0,
                locator=f"synthetic:{i}",
                w=w,
                h=h,
                has_alpha=self._alpha(i),
                group=f"g{i % 4}",
                tags=tags,
                artist=f"artist{i % 4}",
            )
            row.extra["hash"] = text_hash("synthetic", self.seed, i, w, h)
            yield row

    def open(self, locator: str) -> bytes:
        i = int(locator.split(":")[1])
        w, h = self._size(i)
        rng = rng_of("synthetic-pixels", self.seed, i)
        mode = "RGBA" if self._alpha(i) else "RGB"
        base = rng.integers(0, 256, size=(h // 16 + 1, w // 16 + 1, len(mode)))
        img = Image.fromarray(base.astype(np.uint8), mode).resize((w, h))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
