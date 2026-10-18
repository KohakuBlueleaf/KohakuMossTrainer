"""T-LoRA timestep masks for packed batches: one mask group per token axis (joint,
text, image, per-sample), each module reading its axis's group."""

import re

import torch
from lycoris.modules.tlora import (
    clear_timestep_mask,
    get_timestep_mask,
    set_timestep_mask,
)
from torch import Tensor

from kohakumoss.adapters.build.presets import (
    EMBED_HEAD,
    JOINT_ATTN,
    JOINT_MLP,
    MODULATION,
    TEXT_BLOCK,
    TEXT_PROJ,
)

AXES = {"joint": 0, "text": 1, "image": 2, "sample": 3}
AXIS_PATTERNS = (
    ("joint", (JOINT_ATTN, JOINT_MLP)),
    ("text", (TEXT_BLOCK, TEXT_PROJ)),
    ("image", (EMBED_HEAD,)),
    ("sample", (MODULATION,)),
)
MASK_KEYS = {
    "mask_min_rank": "min_rank",
    "mask_alpha": "alpha",
    "mask_max_timestep": "max_timestep",
}


def axis_of(path: str) -> str:
    """The token axis a layer's input rows run over (KeyError for an unknown layer)."""
    for axis, patterns in AXIS_PATTERNS:
        if any(re.search(p, path) for p in patterns):
            return axis
    raise KeyError(f"{path}: no known token axis for a T-LoRA mask")


def rank_mask(
    sigma: Tensor,
    max_rank: int,
    min_rank: int = 1,
    alpha: float = 1.0,
    max_timestep: float = 1000.0,
) -> Tensor:
    """[B, max_rank] mask: sample i keeps the first r_i ranks, r_i from its timestep
    (lycoris.modules.tlora.compute_timestep_mask with t = sigma * max_timestep)."""
    t = sigma.float() * max_timestep
    frac = ((max_timestep - t) / max_timestep).clamp(0, 1) ** alpha
    ranks = (frac * (max_rank - min_rank)).floor().long() + min_rank
    ranks = ranks.clamp(max=max_rank)
    cols = torch.arange(max_rank, device=sigma.device)
    return (cols[None, :] < ranks[:, None]).float()


def token_mask(
    sigma: Tensor,
    seg: Tensor,
    max_rank: int,
    min_rank: int = 1,
    alpha: float = 1.0,
    max_timestep: float = 1000.0,
) -> Tensor:
    """[N, max_rank] mask with one row per token of segment ids `seg`."""
    per_sample = rank_mask(sigma, max_rank, min_rank, alpha, max_timestep)
    return per_sample.index_select(0, seg.long())


def _rows(seq, axis: str, batch: int, device) -> Tensor:
    match axis:
        case "joint":
            return seq.seg_joint
        case "text":
            return seq.seg_txt
        case "image":
            return seq.seg_img
        case _:
            return torch.arange(batch, device=device)


def set_masks(axes: tuple[str, ...], max_rank: int, kwargs: dict, sigma, seq) -> None:
    """Install this step's mask for every axis group in `axes`."""
    per_sample = rank_mask(sigma, max_rank, **kwargs)
    for axis in axes:
        rows = _rows(seq, axis, per_sample.shape[0], per_sample.device)
        set_timestep_mask(per_sample.index_select(0, rows.long()), AXES[axis])


def clear_masks() -> None:
    for group in AXES.values():
        clear_timestep_mask(group)


def saved_masks() -> dict[int, Tensor]:
    """The installed masks, by group (for a save / clear / restore around exports)."""
    out = {}
    for group in AXES.values():
        mask = get_timestep_mask(group)
        if mask is not None:
            out[group] = mask
    return out


def restore_masks(masks: dict[int, Tensor]) -> None:
    clear_masks()
    for group, mask in masks.items():
        set_timestep_mask(mask, group)
