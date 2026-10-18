"""Batching functions over an ordered item stream: greedy tokens, FFD tokens, fixed
count. Each takes `(items, tokens, limits)` (`tokens[i]`: item i's joint token count)
and returns the batches (lists of items)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchLimits:
    """Parameters shared by the batching functions; `max_samples = 0` = no count cap."""

    max_tokens: int = 16384
    batch_size: int = 4
    window: int = 256
    max_samples: int = 0
    drop_last: bool = True

    def cap(self, n: int) -> int:
        """The per-batch sample cap for a stream of `n` items."""
        return self.max_samples or n + 1


def greedy_tokens(items: list, tokens: list[int], limits: BatchLimits) -> list[list]:
    """Walk the stream; a sample that would push the batch past `max_tokens` (or its
    count past `max_samples`) starts the next batch."""
    batches, current, used = [], [], 0
    cap = limits.cap(len(items))
    for item, t in zip(items, tokens):
        if current and (used + t > limits.max_tokens or len(current) >= cap):
            batches.append(current)
            current, used = [], 0
        current.append(item)
        used += t
    if current:
        batches.append(current)
    return batches


def ffd_tokens(items: list, tokens: list[int], limits: BatchLimits) -> list[list]:
    """First-fit-decreasing per window of `window` items into `max_tokens` bins."""
    batches = []
    cap = limits.cap(len(items))
    for start in range(0, len(items), limits.window):
        stop = start + limits.window
        window = sorted(
            zip(tokens[start:stop], range(stop - start), items[start:stop]),
            key=lambda t: (-t[0], t[1]),
        )
        bins: list[list] = []
        room: list[int] = []
        for t, _, item in window:
            for j, free in enumerate(room):
                if t <= free and len(bins[j]) < cap:
                    bins[j].append(item)
                    room[j] -= t
                    break
            else:
                bins.append([item])
                room.append(limits.max_tokens - t)
        batches.extend(bins)
    return batches


def count(items: list, tokens: list[int], limits: BatchLimits) -> list[list]:
    """`batch_size` consecutive samples; the trailing partial batch is kept unless
    `drop_last`."""
    b = limits.batch_size
    stop = len(items) - b + 1 if limits.drop_last else len(items)
    return [items[i : i + b] for i in range(0, max(stop, 0), b)]


BATCHINGS = {"tokens": greedy_tokens, "tokens_ffd": ffd_tokens, "count": count}
