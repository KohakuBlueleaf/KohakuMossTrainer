"""Offline cache passes: every geometry / flip a planner can emit, every caption
variant; keys already stored are skipped (a pass resumes)."""

from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from kohakumoss_data.cache.settings import record
from kohakumoss_data.types import EncodeImages, EncodeText, PlanItem


def check_spec(encode, spec: str, knob: str) -> None:
    """Raise ValueError when an encoder that reports its compute spec (`encoder_spec`)
    does not compute what the cache keys record (`spec`, knob `knob`)."""
    found = getattr(encode, "encoder_spec", None)
    if found is not None and found != spec:
        raise ValueError(
            f"the encoder computes {found!r} but the cache keys record {knob}={spec!r}: "
            f"set {knob} to the encoder's spec"
        )


def latent_items(dataset) -> Iterator[PlanItem]:
    """One item per (live row, flip): the row's planner target with the centre crop.
    Every live row is covered, including rows a given epoch skips (fractional weights,
    drops); tombstoned rows are not."""
    planner = dataset.planner
    if planner.geometry.crop != "center":
        raise ValueError("an offline latent cache needs a centre crop")
    flips = (False, True) if dataset.cfg.flip else (False,)
    catalog = dataset.catalog
    for row_id in catalog.live_ids().tolist():
        target = planner.targets[row_id]
        w, h = int(catalog.w[row_id]), int(catalog.h[row_id])
        box = planner.geometry.center_box(w, h, target, None)
        for flip in flips:
            yield PlanItem(row_id, target, box, flip, 0)


def _decode_chunk(pool: ThreadPoolExecutor, dataset, chunk: list) -> list[Future]:
    """Read the bytes of `chunk`'s items on this thread, decode them on `pool`."""
    return [
        pool.submit(
            dataset.decode,
            dataset.open(item.row_id),
            item.crop_ltrb,
            item.target_hw,
            item.flip,
        )
        for item, _ in chunk
    ]


def build_latents(
    dataset,
    encode: EncodeImages,
    batch_size: int = 8,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Encode and store every missing latent; returns how many were written. Images
    decode on NUM_WORKERS threads, one chunk ahead of the encoder. Raises ValueError
    when the encoder's `encoder_spec` differs from LATENT_ENCODER_SPEC."""
    check_spec(encode, dataset.cfg.latent_encoder_spec, "LATENT_ENCODER_SPEC")
    cache = dataset.latents
    groups: dict[tuple[int, int], list[tuple[PlanItem, str]]] = {}
    for item in latent_items(dataset):
        key = cache.key(dataset.catalog[item.row_id], item)
        if key not in cache.store:
            groups.setdefault(item.target_hw, []).append((item, key))
    chunks = [
        (target, groups[target][i : i + batch_size])
        for target in sorted(groups)
        for i in range(0, len(groups[target]), batch_size)
    ]
    written = 0
    lf = dataset.cfg.latent_factor
    with ThreadPoolExecutor(max(1, dataset.cfg.num_workers)) as pool:
        ahead = _decode_chunk(pool, dataset, chunks[0][1]) if chunks else []
        for k, (target, chunk) in enumerate(chunks):
            images = [future.result() for future in ahead]
            if k + 1 < len(chunks):
                ahead = _decode_chunk(pool, dataset, chunks[k + 1][1])
            h, w = target[0] // lf, target[1] // lf
            packed = encode(images, torch.tensor([[h, w]] * len(chunk)))
            for j, (_, key) in enumerate(chunk):
                cache.put(key, packed[j * h * w : (j + 1) * h * w])
            written += len(chunk)
            if progress is not None:
                progress(written)
    cache.store.flush()
    record(dataset.cfg, "latents")
    return written


def caption_set(dataset) -> list[str]:
    """Every caption variant the planner draws for every live row, plus the empty
    prompt, deduplicated."""
    out = {""}
    variants = dataset.planner.caption_variants
    for row_id in dataset.catalog.live_ids().tolist():
        for variant in range(variants):
            out.add(dataset.caption(row_id, variant))
    return sorted(out)


def build_text(
    dataset,
    encode: EncodeText,
    batch_size: int = 16,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Encode and store every missing caption embedding; returns how many were written.
    Raises ValueError unless `encode.encoder_id` equals the cache's TEXT_ENCODER_ID and
    its `encoder_spec` (when it reports one) TEXT_ENCODER_SPEC."""
    cache = dataset.texts
    cfg = dataset.cfg
    ident = getattr(encode, "encoder_id", None)
    if ident != cfg.text_encoder_id:
        raise ValueError(
            f"text encoder id {ident!r} != cache encoder id {cfg.text_encoder_id!r}: "
            "set TEXT_ENCODER_ID to the encoder that writes this cache"
        )
    check_spec(encode, cfg.text_encoder_spec, "TEXT_ENCODER_SPEC")
    todo = [c for c in caption_set(dataset) if cache.key(c) not in cache.store]
    for i in range(0, len(todo), batch_size):
        chunk = todo[i : i + batch_size]
        for caption, emb in zip(chunk, encode(chunk)):
            cache.put(cache.key(caption), emb)
        if progress is not None:
            progress(i + len(chunk))
    cache.store.write_index()
    cache.store.flush()
    record(dataset.cfg, "text")
    return len(todo)
