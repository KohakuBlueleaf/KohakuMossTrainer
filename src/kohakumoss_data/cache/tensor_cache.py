"""LatentCache / TextCache: mode + encoder identity + key rules over a TensorStore.

`get` / `put` / `lengths` are bound at construction: the store's methods when the
cache is on, no-ops when it is off.
"""

import os

from torch import Tensor

from kohakumoss_data.cache.keys import latent_key, text_key
from kohakumoss_data.cache.store import TensorStore
from kohakumoss_data.types import CatalogRow, PlanItem

MODES = ("off", "offline", "online")


def _miss(key: str) -> Tensor | None:
    return None


def _discard(key: str, value: Tensor) -> None:
    return None


def _no_lengths() -> dict[str, int]:
    return {}


class _Cache:
    def __init__(self, path: str | None, mode: str, encoder_id: str, storage: str):
        if mode not in MODES:
            raise ValueError(f"cache mode must be one of {MODES}, got {mode}")
        if mode != "off" and not path:
            raise ValueError(f"cache mode {mode} needs a path")
        self.mode = mode
        self.encoder_id = encoder_id
        self.storage = storage
        self.store = None
        self.get, self.put, self.lengths = _miss, _discard, _no_lengths
        if mode != "off":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self.store = TensorStore(path, storage)
            self.get, self.put = self.store.get, self.store.put
            self.lengths = self.store.lengths

    @property
    def enabled(self) -> bool:
        return self.store is not None


class LatentCache(_Cache):
    """Packed latent tokens [h*w, C] per (image, geometry, flip)."""

    def __init__(
        self,
        path: str | None,
        mode: str = "off",
        encoder_id: str = "",
        storage: str = "bf16",
        background: str = "white",
    ):
        super().__init__(path, mode, encoder_id, storage)
        self.background = background

    def key(self, row: CatalogRow, item: PlanItem) -> str:
        return latent_key(row, item, self.encoder_id, self.storage, self.background)


class TextCache(_Cache):
    """Unpadded text embeddings [L, D] per rendered caption."""

    def key(self, caption: str) -> str:
        return text_key(caption, self.encoder_id, self.storage)
