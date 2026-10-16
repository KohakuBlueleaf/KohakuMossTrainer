"""Prewarm: tune ops on scratch inputs before training, outside any no_grad / checkpoint
/ graph capture, so live calls only look tuned choices up."""

from collections.abc import Callable

import torch


def attention_lengths(max_len: int, floor: int = 64) -> list[int]:
    """One segment length per power-of-two bucket up to `max_len` (the tuning keys)."""
    out, b = [], floor
    while b < max_len:
        out.append(b)
        b *= 2
    out.append(max_len)
    return out


def prewarm_attention(
    fn: Callable,
    device: torch.device | str,
    dtype: torch.dtype,
    heads: int,
    head_dim: int,
    max_len: int,
    segments: int = 2,
) -> None:
    """Run `fn` (an `attention_varlen` implementation) forward + backward once per
    length bucket up to `max_len`, on its own leaves under `enable_grad`."""
    device = torch.device(device)
    for length in attention_lengths(max_len):
        n = length * segments
        cu = torch.arange(0, n + 1, length, device=device, dtype=torch.int32)
        with torch.enable_grad():
            q, k, v = (
                torch.randn(
                    n, heads, head_dim, device=device, dtype=dtype
                ).requires_grad_()
                for _ in range(3)
            )
            fn(q, k, v, cu, length, head_dim**-0.5).sum().backward()
