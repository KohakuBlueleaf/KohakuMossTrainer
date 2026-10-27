"""Data-side types."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from torch import Tensor

EncodeImages = Callable[[Sequence[np.ndarray], Tensor], Tensor]
"""uint8 HWC images + target hw [B, 2] (latent grid) -> packed latents [N_img, C]."""

EncodeText = Callable[[Sequence[str]], list[Tensor]]
"""prompts -> one unpadded [L_i, text_dim] embedding per prompt."""


@dataclass
class CatalogRow:
    """One image record; `weight` generalises repeats."""

    id: int
    locator: str
    w: int
    h: int
    has_alpha: bool = False
    group: str = ""
    weight: float = 1.0
    tags: list[str] | None = None
    caption_nl: str | None = None
    artist: str | None = None
    rating: str | None = None
    json: dict | None = None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PlanItem:
    """One sample of a planned batch: source row, target pixel size (h, w), crop box,
    flip, caption variant, and its over-long fit (`text_limit` > 0: first text tokens
    kept; `img_keep` > 0: image tokens kept, drawn by `keep_seed`; 0 = all kept)."""

    row_id: int
    target_hw: tuple[int, int]
    crop_ltrb: tuple[int, int, int, int]
    flip: bool = False
    caption_variant: int = 0
    text_limit: int = 0
    img_keep: int = 0
    keep_seed: int = 0


@dataclass(frozen=True)
class RowGeometry:
    """A row's stored size (w, h), its training target (h, w) and the centre crop box
    (l, t, r, b) in stored pixels the target is resized from."""

    w: int
    h: int
    target_hw: tuple[int, int]
    crop_ltrb: tuple[int, int, int, int]


@dataclass
class BatchPlan:
    """A whole batch decided by a planner; a pure function of (seed, rank, index)."""

    index: int
    items: list[PlanItem]
    token_budget: int = 0


def _pinned(value):
    """`value` in pinned memory when it is a tensor, else as is."""
    if isinstance(value, Tensor) and not value.is_pinned():
        return value.pin_memory()
    return value


@dataclass
class HostBatch:
    """Packed batch on the host (pinned when a GPU consumes it); image tokens are
    sample-major, row-major over the latent grid `hw` [B, 2]. Fields: see
    docs/concepts/data.md#hostbatch."""

    ids: Tensor
    hw: Tensor
    text_len: Tensor
    text_drop: Tensor
    weight: Tensor
    img_keep: list[Tensor | None] | None = None
    latents: Tensor | None = None
    text: Tensor | None = None
    pixels: list[np.ndarray] | None = None
    prompts: list[str] | None = None
    loss_mask: Tensor | None = None
    index: int = -1
    rank: int = 0
    latent_keys: list[str] = field(default_factory=list)
    text_keys: list[str] = field(default_factory=list)
    text_limit: list[int] = field(default_factory=list)
    layout: object | None = None

    def pin(self) -> "HostBatch":
        """Pin every tensor field, every tensor of a list field and the `layout` (its
        `pin_memory()`) in place; returns self. See docs/concepts/data.md#hostbatch."""
        for name, value in self.__dict__.items():
            match value:
                case Tensor():
                    setattr(self, name, _pinned(value))
                case list():
                    setattr(self, name, [_pinned(v) for v in value])
                case _ if hasattr(value, "pin_memory"):
                    setattr(self, name, value.pin_memory())
        return self

    def pin_memory(self) -> "HostBatch":
        """DataLoader pin-memory hook."""
        return self.pin()
