"""Prewarm: tune ops on scratch inputs before training, outside any no_grad / checkpoint
/ graph capture, so live calls only look tuned choices up."""

import time
from collections.abc import Callable
from functools import partial

import torch

from kohakumoss_common.log import get_logger, hl
from kohakumoss_kernels.hardware.current import rng_kept
from kohakumoss_kernels.registry import Bound, exercise
from kohakumoss_kernels.tuning.buckets import LENGTH_FLOOR, SEGMENT_CAP, pow2_buckets

log = get_logger(__name__)
# one segment length per power-of-two bucket up to `max_len` (the tuning keys)
attention_lengths = partial(pow2_buckets, floor=LENGTH_FLOOR)


def reachable_segments(
    length: int, segments: int, max_tokens: int, shortest: int
) -> int:
    """The most segments a batch whose longest segment is `length` holds: `segments`
    (the batches' most samples), fewer when `length` plus `shortest`-token segments
    pass `max_tokens` (0: no cap; `shortest` 0: `length`); one at least."""
    if not max_tokens:
        return max(1, segments)
    short = min(shortest or length, length)
    return max(1, min(segments, 1 + (max_tokens - length) // short))


def segment_counts(
    length: int, segments: int, max_tokens: int, shortest: int
) -> list[int]:
    """One segment count per segment-count bucket (1, 2, 4, then SEGMENT_CAP and
    above) that a batch with longest segment `length` reaches (`reachable_segments`)."""
    most = reachable_segments(length, segments, max_tokens, shortest)
    out, b = [], 1
    while True:
        out.append(min(b, most))
        if b >= most or b >= SEGMENT_CAP:
            return out
        b *= 2


def prewarm_attention(
    fn: Callable,
    device: torch.device | str,
    dtype: torch.dtype,
    heads: int,
    head_dim: int,
    max_len: int,
    segments: int = 1,
    backward: bool = True,
    max_tokens: int = 0,
    shortest: int = 0,
    label: str = "attention_varlen",
) -> None:
    """Run `fn` (an `attention_varlen` implementation) forward + backward once per
    length bucket up to `max_len` and per segment-count bucket the planned batches
    reach there (`segment_counts` over at most `segments` samples per batch,
    `max_tokens` tokens per batch and samples of `shortest` tokens at least): one
    segment of the bucket's length and the rest `shortest` long, on its own leaves
    under `enable_grad`; with `backward=False` (inference) forward only under
    `no_grad`. A `Bound` op whose backend fails here falls back before the first live
    call. `label` names the call site in the log."""
    device = torch.device(device)
    grad, run = (torch.enable_grad, _step) if backward else (torch.no_grad, _forward)
    lengths = attention_lengths(max_len)
    start = time.perf_counter()
    for i, length in enumerate(lengths, start=1):
        count, tokens = f"{i}/{len(lengths)}", f"{length} tokens"
        log.info(f"tuning {label} {hl(count)} ({hl(tokens)})")
        short = min(shortest or length, length)
        for segs in segment_counts(length, segments, max_tokens, shortest):
            sizes = [length] + [short] * (segs - 1)
            with rng_kept([device]):
                _call(fn, run, grad, device, dtype, heads, head_dim, sizes)
    took = time.perf_counter() - start
    log.info(f"prewarm {label} on {device}: done in {hl(f'{took:.2f}s')}")


def _call(fn, run, grad, device, dtype, heads, head_dim, sizes: list[int]) -> None:
    """One prewarm call over segments of `sizes` tokens (the first the longest)."""
    n = sum(sizes)
    cu = torch.tensor([0, *sizes], device=device, dtype=torch.int64).cumsum(0)
    cu = cu.to(torch.int32)
    with grad():
        q, k, v = (
            torch.randn(n, heads, head_dim, device=device, dtype=dtype)
            for _ in range(3)
        )
        step = partial(run, q=q, k=k, v=v, cu=cu, length=sizes[0])
        if isinstance(fn, Bound):
            exercise(fn, step)
        else:
            step(fn)


def _step(op: Callable, q, k, v, cu, length: int) -> None:
    leaves = [t.detach().requires_grad_() for t in (q, k, v)]
    op(*leaves, cu, length, q.shape[-1] ** -0.5).sum().backward()


def _forward(op: Callable, q, k, v, cu, length: int) -> None:
    op(q, k, v, cu, length, q.shape[-1] ** -0.5)
