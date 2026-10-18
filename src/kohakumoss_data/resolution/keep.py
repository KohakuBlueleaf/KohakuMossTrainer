"""The kept image tokens of a plan item fitted by `drop_tokens`."""

import torch
from torch import Tensor

from kohakumoss_data.seeds import rng_of
from kohakumoss_data.types import PlanItem


def keep_indices(item: PlanItem, n_tokens: int) -> Tensor | None:
    """Sorted int64 indices of the kept image tokens of `item`, None when all are kept."""
    if not item.img_keep:
        return None
    rng = rng_of("img-keep", item.keep_seed)
    picked = rng.choice(n_tokens, size=item.img_keep, replace=False)
    return torch.from_numpy(picked.astype("int64")).sort().values
