"""Offline cache passes: every geometry/flip a planner can emit, every caption variant.

Both passes call only the `EncodeImages` / `EncodeText` callables and skip keys that
are already stored, so an interrupted pass resumes where it stopped.
"""

from collections.abc import Callable, Iterator

import torch

from kohakumoss_data.types import EncodeImages, EncodeText, PlanItem


def latent_items(dataset) -> Iterator[PlanItem]:
    """One item per (row, flip): the row's planner target with the centre crop. Every
    row is covered, including rows a given epoch skips (fractional weights, drops)."""
    planner = dataset.planner
    if planner.geometry.crop != "center":
        raise ValueError("an offline latent cache needs a centre crop")
    flips = (False, True) if dataset.cfg.flip else (False,)
    for row_id, row in enumerate(dataset.catalog.rows):
        target = planner.targets[row_id]
        box = planner.geometry.center_box(row.w, row.h, target, None)
        for flip in flips:
            yield PlanItem(row_id, target, box, flip, 0)


def build_latents(
    dataset,
    encode: EncodeImages,
    batch_size: int = 8,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Encode and store every missing latent; returns how many were written."""
    cache = dataset.latents
    groups: dict[tuple[int, int], list[PlanItem]] = {}
    for item in latent_items(dataset):
        key = cache.key(dataset.catalog[item.row_id], item)
        if key not in cache.store:
            groups.setdefault(item.target_hw, []).append(item)
    written = 0
    lf = dataset.cfg.latent_factor
    for target in sorted(groups):
        items = groups[target]
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            images = [dataset.pixels(it) for it in chunk]
            hw = torch.tensor([[target[0] // lf, target[1] // lf]] * len(chunk))
            packed = encode(images, hw)
            per = (target[0] // lf) * (target[1] // lf)
            for k, item in enumerate(chunk):
                row = dataset.catalog[item.row_id]
                cache.put(cache.key(row, item), packed[k * per : (k + 1) * per])
            written += len(chunk)
            if progress is not None:
                progress(written)
    cache.store.flush()
    return written


def caption_set(dataset) -> list[str]:
    """Every caption variant of every row, plus the empty prompt, deduplicated."""
    out = {""}
    for row_id in range(len(dataset.catalog)):
        for variant in range(dataset.cfg.caption_variants):
            out.add(dataset.caption(row_id, variant))
    return sorted(out)


def build_text(
    dataset,
    encode: EncodeText,
    batch_size: int = 16,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Encode and store every missing caption embedding; returns how many were written."""
    cache = dataset.texts
    todo = [c for c in caption_set(dataset) if cache.key(c) not in cache.store]
    for i in range(0, len(todo), batch_size):
        chunk = todo[i : i + batch_size]
        for caption, emb in zip(chunk, encode(chunk)):
            cache.put(cache.key(caption), emb)
        if progress is not None:
            progress(i + len(chunk))
    cache.store.write_index()
    cache.store.flush()
    return len(todo)
