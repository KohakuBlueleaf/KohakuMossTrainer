"""Variable-length attention reference: one SDPA call per segment, no cross-segment keys."""

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["attention_varlen"]


def attention_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    scale: float,
) -> Tensor:
    """q, k, v [N, H, Dh] packed by `cu_seqlens` [B+1]; full attention inside segments."""
    bounds = cu_seqlens.tolist()
    out = torch.empty_like(q)
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end == start:
            continue
        qs, ks, vs = (t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        o = F.scaled_dot_product_attention(qs, ks, vs, scale=scale)
        out[start:end] = o.squeeze(0).transpose(0, 1)
    return out
