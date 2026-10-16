"""T-LoRA timestep masks for packed batches (one mask row per token)."""

import torch
from lycoris.modules.tlora import set_timestep_mask
from torch import Tensor


def token_mask(
    sigma: Tensor,
    seg: Tensor,
    max_rank: int,
    min_rank: int = 1,
    alpha: float = 1.0,
    max_timestep: float = 1000.0,
) -> Tensor:
    """[N, max_rank] mask: sample i keeps the first r_i ranks, r_i from its timestep.

    Same rule as lycoris.modules.tlora.compute_timestep_mask with t = σ·max_timestep.
    """
    t = sigma.float() * max_timestep
    frac = ((max_timestep - t) / max_timestep).clamp(0, 1) ** alpha
    ranks = (frac * (max_rank - min_rank)).floor().long() + min_rank
    ranks = ranks.clamp(max=max_rank)
    cols = torch.arange(max_rank, device=sigma.device)
    per_sample = (cols[None, :] < ranks[:, None]).float()
    return per_sample.index_select(0, seg.long())


def set_token_mask(sigma: Tensor, seg: Tensor, max_rank: int, **kwargs) -> None:
    """Install the per-token mask every T-LoRA module of group 0 reads this step."""
    set_timestep_mask(token_mask(sigma, seg, max_rank, **kwargs), 0)
