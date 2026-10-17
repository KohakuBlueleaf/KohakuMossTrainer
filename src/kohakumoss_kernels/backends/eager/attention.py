"""Variable-length attention reference: one SDPA call per segment, no cross-segment keys.

Segment bounds come from a host copy cached per `cu_seqlens` tensor (registered by the
layout builder, or read back once); see docs/internals/kernel-dev.md.
"""

import weakref

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["attention_varlen"]

_BOUNDS: dict[int, tuple[weakref.ref, int, list[int]]] = {}


def _forget(key: int):
    return lambda _ref: _BOUNDS.pop(key, None)


def remember_bounds(cu_seqlens: Tensor, bounds) -> None:
    """Register the host values of `cu_seqlens`, kept while the tensor lives and its
    version counter is unchanged; inference tensors have none and are not cached."""
    if cu_seqlens.is_inference():
        return
    key = id(cu_seqlens)
    _BOUNDS[key] = (
        weakref.ref(cu_seqlens, _forget(key)),
        cu_seqlens._version,
        [int(b) for b in bounds],
    )


def host_bounds(cu_seqlens: Tensor) -> list[int]:
    """The host values of `cu_seqlens`, read back (one sync) only on first sight of a
    versioned tensor and on every call for an inference tensor."""
    if cu_seqlens.is_inference():
        return cu_seqlens.tolist()
    hit = _BOUNDS.get(id(cu_seqlens))
    if hit is not None and hit[0]() is cu_seqlens and hit[1] == cu_seqlens._version:
        return hit[2]
    bounds = cu_seqlens.tolist()
    remember_bounds(cu_seqlens, bounds)
    return bounds


def attention_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    scale: float,
) -> Tensor:
    """q, k, v [N, H, Dh] packed by `cu_seqlens` [B+1]; full attention inside segments."""
    bounds = host_bounds(cu_seqlens)
    out = torch.empty_like(q)
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end == start:
            continue
        qs, ks, vs = (t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        o = F.scaled_dot_product_attention(qs, ks, vs, scale=scale)
        out[start:end] = o.squeeze(0).transpose(0, 1)
    return out
