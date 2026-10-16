"""Batching rules over an ordered item stream: greedy tokens, FFD tokens, fixed count.

Every rule takes `(items, tokens, rule)` where `tokens[i]` is item i's joint token
count and returns the list of batches (lists of items). `max_samples = 0` means no
count cap; it is turned into a cap above any batch size, so no rule branches on it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchRule:
    """Parameters shared by the batching rules."""

    max_tokens: int = 16384
    batch_size: int = 4
    window: int = 256
    max_samples: int = 0
    drop_last: bool = True

    def cap(self, n: int) -> int:
        return self.max_samples or n + 1


def greedy_tokens(items: list, tokens: list[int], rule: BatchRule) -> list[list]:
    """Walk the stream; a sample that would push the batch past `max_tokens` (or its
    count past `max_samples`) starts the next batch."""
    batches, current, used = [], [], 0
    cap = rule.cap(len(items))
    for item, t in zip(items, tokens):
        if current and (used + t > rule.max_tokens or len(current) >= cap):
            batches.append(current)
            current, used = [], 0
        current.append(item)
        used += t
    if current:
        batches.append(current)
    return batches


def ffd_tokens(items: list, tokens: list[int], rule: BatchRule) -> list[list]:
    """First-fit-decreasing per window of `window` items into `max_tokens` bins."""
    batches = []
    cap = rule.cap(len(items))
    for start in range(0, len(items), rule.window):
        stop = start + rule.window
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
                room.append(rule.max_tokens - t)
        batches.extend(bins)
    return batches


def count(items: list, tokens: list[int], rule: BatchRule) -> list[list]:
    """`batch_size` consecutive samples; the trailing partial batch is kept unless
    `drop_last`."""
    b = rule.batch_size
    stop = len(items) - b + 1 if rule.drop_last else len(items)
    return [items[i : i + b] for i in range(0, max(stop, 0), b)]


BATCHINGS = {"tokens": greedy_tokens, "tokens_ffd": ffd_tokens, "count": count}
