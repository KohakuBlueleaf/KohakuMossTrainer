"""Rectified-flow noising and the velocity target on packed image tokens."""

from torch import Tensor

from kohakumoss.types.seqinfo import SeqInfo


def noise_latents(
    z0: Tensor, eps: Tensor, sigma: Tensor, seq: SeqInfo
) -> tuple[Tensor, Tensor]:
    """z_sigma = (1 - sigma) z0 + sigma eps and target eps - z0; sigma [B] per sample."""
    s = sigma.to(z0.dtype).index_select(0, seq.seg_img).unsqueeze(-1)
    return z0 + s * (eps - z0), eps - z0
