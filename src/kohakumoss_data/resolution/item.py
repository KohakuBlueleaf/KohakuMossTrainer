"""MossPlanItem: a PlanItem plus the over-long fit decisions (text limit, image keep)."""

from dataclasses import dataclass

import torch
from torch import Tensor

from kohakumoss_data.seeds import rng_of
from kohakumoss_data.types import PlanItem


@dataclass(frozen=True)
class MossPlanItem(PlanItem):
    """`text_limit` > 0 keeps only the first `text_limit` text tokens; `img_keep` > 0
    keeps that many image tokens, chosen by `keep_seed` (0 = all kept)."""

    text_limit: int = 0
    img_keep: int = 0
    keep_seed: int = 0


def keep_indices(item: PlanItem, n_tokens: int) -> Tensor | None:
    """Sorted int64 indices of the kept image tokens of `item`, None when all are kept."""
    k = getattr(item, "img_keep", 0)
    if not k:
        return None
    rng = rng_of("img-keep", item.keep_seed)
    picked = rng.choice(n_tokens, size=k, replace=False)
    return torch.from_numpy(picked.astype("int64")).sort().values
