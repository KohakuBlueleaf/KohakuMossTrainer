"""Data-side types and protocols. No model code, no torch.distributed."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch
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
    """One sample of a planned batch: source row, target pixel size, crop box, flip."""

    row_id: int
    target_hw: tuple[int, int]
    crop_ltrb: tuple[int, int, int, int]
    flip: bool = False
    caption_variant: int = 0


@dataclass
class BatchPlan:
    """A whole batch decided by a planner; a pure function of (seed, rank, index)."""

    index: int
    items: list[PlanItem]
    token_budget: int = 0


@dataclass
class HostBatch:
    """Packed batch on the host (pinned when a GPU consumes it).

    Image tokens are sample-major and row-major over the latent grid `hw` [B, 2].
    `img_keep[i]` lists the kept grid indices of sample i after random token drop
    (None = all kept); cached `latents` then hold only the kept tokens, while `pixels`
    are the full image (the consumer encodes, then gathers `img_keep`).
    `text_limit[i]` > 0 caps sample i's text tokens: cached `text` is already capped
    and `text_len` final; with `prompts` the consumer keeps the first `text_limit`
    rows after encoding and fills `text_len`.
    `index` / `rank` locate the plan; `latent_keys` / `text_keys` are each sample's
    cache keys (empty when that cache is off), used by online write-back.
    """

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

    def pin(self) -> "HostBatch":
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and not value.is_pinned():
                setattr(self, name, value.pin_memory())
        return self

    def pin_memory(self) -> "HostBatch":
        """DataLoader pin-memory hook."""
        return self.pin()


class Source(Protocol):
    """A dataset backend that can enumerate records and open their image bytes."""

    def scan(self) -> Iterator[CatalogRow]: ...

    def open(self, locator: str) -> bytes: ...


class Planner(Protocol):
    """Decides batch `index` for `rank` of `world` from the catalog."""

    def plan(self, index: int, rank: int, world: int) -> BatchPlan: ...


class CaptionTemplate(Protocol):
    """Renders a caption string for a row with a seeded generator."""

    def render(self, row: CatalogRow, rng: np.random.Generator) -> str: ...


class Cache(Protocol):
    """Content-addressed tensor cache (latents or text embeddings)."""

    def key(self, row: CatalogRow, item: PlanItem) -> str: ...

    def get(self, key: str) -> Tensor | None: ...

    def put(self, key: str, value: Tensor) -> None: ...


def empty_meta() -> dict[str, Any]:
    return {"dtype": str(torch.bfloat16)}
