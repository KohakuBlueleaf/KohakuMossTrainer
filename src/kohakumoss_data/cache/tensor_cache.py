"""LatentCache / TextCache: mode + encoder identity + cache keys over a TensorStore;
`get` / `put` / `lengths` are the store's when the cache is on, no-ops when off."""

import os

import torch
from torch import Tensor

from kohakumoss_common.calls import nothing
from kohakumoss_data.cache.keys import latent_key, text_key
from kohakumoss_data.cache.store import LengthTable, TensorStore, stored_value
from kohakumoss_data.types import CatalogRow, PlanItem

MODES = ("off", "offline", "online")


def _no_lengths() -> LengthTable:
    return LengthTable()


class _Cache:
    """Mode, encoder id and storage dtype shared by both caches."""

    def __init__(self, path: str | None, mode: str, encoder_id: str, storage: str):
        if mode not in MODES:
            raise ValueError(f"cache mode must be one of {MODES}, got {mode}")
        if mode != "off" and not path:
            raise ValueError(f"cache mode {mode} needs a path")
        self.mode = mode
        self.encoder_id = encoder_id
        self.storage = storage
        self.store = None
        self.get, self.put, self.lengths = nothing, nothing, _no_lengths
        if mode != "off":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self.store = TensorStore(path, storage)
            self.get, self.put = self.store.get, self.store.put
            self.lengths = self.store.lengths

    @property
    def enabled(self) -> bool:
        return self.store is not None


class LatentCache(_Cache):
    """Packed latent tokens [h*w, C] per (image, geometry, flip, decode mode)."""

    def __init__(
        self,
        path: str | None,
        mode: str = "off",
        encoder_id: str = "",
        storage: str = "bf16",
        background: str = "white",
        resample: str = "lanczos",
        draw: str = "mode",
    ):
        super().__init__(path, mode, encoder_id, storage)
        self.background = background
        self.resample = resample
        self.draw = draw

    def key(self, row: CatalogRow, item: PlanItem) -> str:
        """The cache key of `item` of `row`."""
        return latent_key(
            row,
            item,
            self.encoder_id,
            self.storage,
            self.background,
            self.resample,
            self.draw,
        )

    def as_cached(
        self, full: Tensor, hw: Tensor, out_dtype: torch.dtype = torch.bfloat16
    ) -> Tensor:
        """Packed full-grid latents [sum(h*w), C] rounded per sample exactly as storing
        and reading them back would."""
        sizes = (hw[:, 0].long() * hw[:, 1].long()).tolist()
        parts = full.split(sizes, 0)
        return torch.cat([stored_value(p, self.storage, out_dtype) for p in parts], 0)


class TextCache(_Cache):
    """Unpadded text embeddings [L, D] per rendered caption."""

    def key(self, caption: str) -> str:
        """The cache key of a rendered caption."""
        return text_key(caption, self.encoder_id, self.storage)

    def as_cached(
        self, parts: list[Tensor], out_dtype: torch.dtype = torch.bfloat16
    ) -> list[Tensor]:
        """Per-caption embeddings rounded exactly as storing and reading them back would."""
        return [stored_value(p, self.storage, out_dtype) for p in parts]
