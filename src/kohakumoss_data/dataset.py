"""MossDataset: sources + catalog + captions + planner + caches -> HostBatch per index;
the cache modes, text lengths, text dropout, decoder and write-back are bound at
construction."""

import math
import os
from functools import partial

import numpy as np
import torch

from kohakumoss_common.calls import nothing
from kohakumoss_common.log import get_logger
from kohakumoss_common.registry import build
from kohakumoss_data.cache.settings import offline_miss, record
from kohakumoss_data.cache.store import LengthTable
from kohakumoss_data.cache.tensor_cache import LatentCache, TextCache
from kohakumoss_data.caption.template import bind_template, build_template
from kohakumoss_data.catalog.vault import is_catalog_vault
from kohakumoss_data.config import (
    TOKEN_BUDGETS,
    DataConfig,
    catalog_cache_dir,
    encoder_key_id,
    resolved_cache_dir,
)
from kohakumoss_data.format.catalog import Catalog, read_catalog
from kohakumoss_data.loader.decode import Decoder
from kohakumoss_data.registry import PLANNERS, SOURCES
from kohakumoss_data.resolution.geometry import Geometry
from kohakumoss_data.resolution.keep import keep_indices
from kohakumoss_data.scan.paths import scan_path
from kohakumoss_data.scan.runner import scan_catalog
from kohakumoss_data.scan.walk import rows_of
from kohakumoss_data.seeds import rng_of
from kohakumoss_data.types import HostBatch, PlanItem, RowGeometry

DATASET_KEYS = ("weight", "caption", "loss_weight", "catalog", "import_to")
log = get_logger(__name__)


def source_kwargs(entry: dict) -> dict:
    """A dataset entry without its dataset-level keys: `source` + its arguments."""
    return {k: v for k, v in entry.items() if k not in DATASET_KEYS}


def build_source(entry: dict):
    """A source from a dataset entry (the dataset-level keys are stripped)."""
    return build(source_kwargs(entry), SOURCES, key="source")


def scan_source(entry: dict, source, cfg: DataConfig, writer: bool = True) -> Catalog:
    """The catalog of one dataset entry: its prebuilt `catalog` file, the source's own
    catalog (moss_shard), or a scan through the entry's catalog file in the catalog
    cache dir (this process writes it when `writer`; uncached without a cache dir)."""
    path = entry.get("catalog")
    if path and is_catalog_vault(path):
        return read_catalog(path)
    if path and os.path.exists(path) and getattr(source, "walk", None) is not None:
        log.warning(
            f"catalog {path} has another layout: the dataset is scanned instead"
        )
    elif path and os.path.exists(path):
        return read_catalog(path)
    own = getattr(source, "catalog", None)
    if own is not None:
        return own()
    if getattr(source, "walk", None) is None:
        return Catalog(list(source.scan()))
    kwargs = source_kwargs(entry)
    cache_dir = catalog_cache_dir(cfg)
    target = scan_path(cache_dir, kwargs) if cache_dir else None
    return scan_catalog(
        source,
        target,
        writer,
        cfg.data_rescan,
        cfg.data_scan_workers,
        cfg.data_wait,
        kwargs["source"],
    )


def first_captions(
    entry: dict, caption: dict, n: int = 1, seed: int = 0, limit: int = 64
) -> list[str]:
    """The first `n` non-empty rendered captions (variant 0) among the first `limit`
    rows of dataset entry `entry`, reading only those rows."""
    source = build_source(entry)
    render = bind_template(build_template(entry.get("caption", caption)), source)
    if getattr(source, "catalog", None) is not None:
        catalog = source.catalog()
        rows = (catalog[i] for i in range(len(catalog)))
    elif getattr(source, "walk", None) is not None:
        rows = rows_of(source.walk())
    else:
        rows = source.scan()
    out = []
    for i, row in enumerate(rows):
        text = render(row, rng_of("caption", seed, i, 0), i)
        if text:
            out.append(text)
        if len(out) == n or i + 1 == limit:
            break
    return out


def all_chars(catalog: Catalog) -> np.ndarray:
    """Every caption-source char count summed (templates without `estimate_chars`)."""
    return sum(catalog.chars.values())


def text_estimate(cfg: DataConfig, catalog: Catalog, templates: list) -> np.ndarray:
    """Estimated text tokens per row (int64 [N]) from the caption-source char columns:
    each row's dataset template estimates its caption length, then
    ceil(chars / text_chars_per_token) + overhead, capped at max_text_tokens."""
    chars = np.zeros(len(catalog), dtype=np.int64)
    for k, template in enumerate(templates):
        rows = catalog.source == k
        estimate = getattr(template, "estimate_chars", all_chars)
        chars[rows] = estimate(catalog)[rows]
    per = np.float32(cfg.text_chars_per_token)
    tokens = np.ceil(chars.astype(np.float32) / per).astype(np.int64)
    return np.minimum(tokens + cfg.text_overhead_tokens, cfg.max_text_tokens)


def text_chars_budget(cfg: DataConfig) -> int:
    """The caption characters `max_text_tokens` holds under the estimate the planner uses
    (`text_chars_per_token`, `text_overhead_tokens`)."""
    tokens = max(cfg.max_text_tokens - cfg.text_overhead_tokens, 1)
    return int(tokens * cfg.text_chars_per_token)


def draw_drop(seed: int, p: float, index: int, rank: int, slot: int) -> bool:
    """Seeded text-dropout draw for one sample slot."""
    return bool(rng_of("text-drop", seed, index, rank, slot).random() < p)


def never_drop(index: int, rank: int, slot: int) -> bool:
    return False


def _first_missing(found: list) -> int:
    return next(i for i, t in enumerate(found) if t is None)


class MossDataset:
    """Pure function `materialize(index, rank, world) -> HostBatch`; picklable.
    `writer`: this process updates the catalog files (the single writer; a multi-rank
    trainer passes it on rank 0 only, the others wait and read)."""

    def __init__(self, cfg: DataConfig, writer: bool = True):
        if not cfg.datasets:
            raise ValueError("DATASETS is empty")
        self.cfg = cfg
        self.sources = [build_source(entry) for entry in cfg.datasets]
        parts = [
            scan_source(entry, src, cfg, writer)
            for entry, src in zip(cfg.datasets, self.sources)
        ]
        weights = [float(entry.get("weight", 1.0)) for entry in cfg.datasets]
        self.catalog = Catalog.concat(parts, weights)
        self.templates = [
            build_template(entry.get("caption", cfg.caption)) for entry in cfg.datasets
        ]
        self._text_est = text_estimate(cfg, self.catalog, self.templates)
        self.loss_weight = np.array(
            [float(entry.get("loss_weight", 1.0)) for entry in cfg.datasets]
        )
        cache_dir = resolved_cache_dir(cfg)
        self.latents = LatentCache(
            os.path.join(cache_dir, "latents.db") if cache_dir else None,
            cfg.latent_cache,
            encoder_key_id(cfg.latent_encoder_id, cfg.latent_encoder_spec),
            cfg.cache_dtype,
            str(cfg.background),
            cfg.resample,
            cfg.latent_draw,
        )
        self.texts = TextCache(
            os.path.join(cache_dir, "text.db") if cache_dir else None,
            cfg.text_cache,
            encoder_key_id(cfg.text_encoder_id, cfg.text_encoder_spec),
            cfg.cache_dtype,
        )
        self.geometry = Geometry(
            area=cfg.target_area,
            step=cfg.size_step,
            aspect_range=tuple(cfg.aspect_range),
            upscale=cfg.upscale,
            crop=cfg.crop,
            latent_factor=cfg.latent_factor,
        )
        self.decode = Decoder(cfg.background, cfg.resample)
        for kind, mode in (("latents", cfg.latent_cache), ("text", cfg.text_cache)):
            if mode == "online":
                record(cfg, kind)
        offline = cfg.text_cache == "offline"
        self._text_lengths = self.texts.lengths() if offline else LengthTable()
        self._bind()
        self.planner = self._planner()
        self.renderers = [
            bind_template(t, s, self.row_geometry, text_chars_budget(cfg))
            for t, s in zip(self.templates, self.sources)
        ]

    def _bind(self) -> None:
        cfg = self.cfg
        if cfg.token_budget not in TOKEN_BUDGETS:
            raise ValueError(f"token_budget must be one of {TOKEN_BUDGETS}")
        if cfg.token_budget == "exact" and cfg.text_cache != "offline":
            raise ValueError("token_budget='exact' needs text_cache='offline'")
        cls = type(self)
        self._text_len = {
            ("estimate", "off"): cls._text_len_columns,
            ("estimate", "online"): cls._text_len_columns,
            ("estimate", "offline"): cls._text_len_columns,
            ("render", "off"): cls._text_len_estimate,
            ("render", "online"): cls._text_len_estimate,
            ("render", "offline"): cls._text_len_cached,
            ("exact", "offline"): cls._text_len_exact,
        }[(cfg.token_budget, cfg.text_cache)]
        self._drop = (
            partial(draw_drop, cfg.seed, cfg.text_drop) if cfg.text_drop else never_drop
        )
        self._fill_latents = {
            "off": cls._latents_decode,
            "offline": cls._latents_offline,
            "online": cls._latents_online,
        }[cfg.latent_cache]
        self._fill_text = {
            "off": cls._text_prompts,
            "offline": cls._text_offline,
            "online": cls._text_online,
        }[cfg.text_cache]
        latent_online = cfg.latent_cache == "online"
        text_online = cfg.text_cache == "online"
        self._put_latents = cls._write_latents if latent_online else nothing
        self._put_text = cls._write_text if text_online else nothing

    def text_len(self, row_id: int, variant: int) -> int:
        """Text tokens of caption `variant` of row `row_id` under the token budget."""
        return self._text_len(self, row_id, variant)

    def _planner(self):
        cfg = self.cfg
        kwargs = dict(
            catalog=self.catalog,
            geometry=self.geometry,
            seed=cfg.seed,
            flip=cfg.flip,
            caption_variants=cfg.caption_variants,
            text_len=self.text_len,
            batching=cfg.batching,
            overlong=cfg.overlong,
            max_tokens=cfg.max_tokens,
            max_samples=cfg.max_samples,
            batch_size=cfg.batch_size,
        )
        kwargs.update(cfg.planner_kwargs)
        return PLANNERS.get(cfg.resolution_mode)(**kwargs)

    def caption(self, row_id: int, variant: int) -> str:
        """Caption variant `variant` of row `row_id` (fixed per seed, row, variant)."""
        row = self.catalog[row_id]
        render = self.renderers[int(self.catalog.source[row_id])]
        return render(row, rng_of("caption", self.cfg.seed, row_id, variant), row_id)

    def row_geometry(self, row_id: int) -> RowGeometry:
        """Row `row_id`'s stored size, planned target and centre crop box."""
        w, h = int(self.catalog.w[row_id]), int(self.catalog.h[row_id])
        target = tuple(self.planner.targets[row_id])
        crop = self.geometry.center_box(w, h, target, None)
        return RowGeometry(w, h, target, crop)

    def _live_text(self) -> np.ndarray:
        return self._text_est[np.asarray(self.catalog.live, dtype=bool)]

    def text_tokens_max(self) -> int:
        """The longest caption of the live rows in estimated text tokens (the estimate
        columns: chars / text_chars_per_token + overhead, capped at max_text_tokens)."""
        est = self._live_text()
        return int(est.max()) if len(est) else 0

    def text_tokens_min(self) -> int:
        """The shortest text of the live rows in estimated text tokens (the estimate
        columns the token budget packs by), the dropped caption's ("") included when
        caption dropout is on; 0 for an empty catalog."""
        est = self._live_text()
        mins = [int(est.min())] if len(est) else []
        if self.cfg.text_drop > 0:
            mins.append(self.estimate_text_tokens(""))
        return min(mins, default=0)

    def image_tokens_min(self) -> int:
        """The smallest planned image grid of the live rows in latent tokens (the
        resolution planner's per-row targets; 0 for an empty catalog)."""
        live = np.asarray(self.catalog.live, dtype=bool)
        grids = [
            self.geometry.tokens(t) for t, ok in zip(self.planner.targets, live) if ok
        ]
        return min(grids, default=0)

    def estimate_text_tokens(self, caption: str) -> int:
        """Estimated text tokens of a rendered caption."""
        n = math.ceil(len(caption) / self.cfg.text_chars_per_token)
        return min(n + self.cfg.text_overhead_tokens, self.cfg.max_text_tokens)

    def _text_len_columns(self, row_id: int, variant: int) -> int:
        return int(self._text_est[row_id])

    def _text_len_estimate(self, row_id: int, variant: int) -> int:
        return self.estimate_text_tokens(self.caption(row_id, variant))

    def _text_len_cached(self, row_id: int, variant: int) -> int:
        caption = self.caption(row_id, variant)
        known = self._text_lengths.get(self.texts.key(caption))
        return self.estimate_text_tokens(caption) if known is None else known

    def _text_len_exact(self, row_id: int, variant: int) -> int:
        caption = self.caption(row_id, variant)
        known = self._text_lengths.get(self.texts.key(caption))
        if known is None:
            raise KeyError(
                f"token_budget='exact': caption of row {row_id} variant {variant} is "
                "not in the text cache; run scripts/data/cache_text.py"
            )
        return known

    def open(self, row_id: int) -> bytes:
        """The encoded image bytes of row `row_id`."""
        source = self.sources[int(self.catalog.source[row_id])]
        return source.open(self.catalog.locator(row_id))

    def pixels(self, item: PlanItem) -> np.ndarray:
        """uint8 [H, W, 3] of `item`: decoded, cropped, resized, flipped."""
        return self.decode(
            self.open(item.row_id), item.crop_ltrb, item.target_hw, item.flip
        )

    def materialize(self, index: int, rank: int = 0, world: int = 1) -> HostBatch:
        """The batch `index` of `rank`: captions, text dropout, cache lookups, decode."""
        items = self.planner.plan(index, rank, world).items
        lf = self.cfg.latent_factor
        drops = [self._drop(index, rank, slot) for slot in range(len(items))]
        captions = [
            "" if drop else self.caption(it.row_id, it.caption_variant)
            for it, drop in zip(items, drops)
        ]
        ids = [it.row_id for it in items]
        grid = [(it.target_hw[0] // lf, it.target_hw[1] // lf) for it in items]
        batch = HostBatch(
            ids=torch.tensor(ids, dtype=torch.int64),
            hw=torch.tensor(grid, dtype=torch.int32),
            text_len=torch.zeros(len(items), dtype=torch.int32),
            text_drop=torch.tensor(drops, dtype=torch.bool),
            weight=torch.tensor(
                self.loss_weight[self.catalog.source[ids]], dtype=torch.float32
            ),
            img_keep=[keep_indices(it, h * w) for it, (h, w) in zip(items, grid)],
            index=index,
            rank=rank,
            text_limit=[it.text_limit for it in items],
        )
        self._fill_latents(self, batch, items)
        self._fill_text(self, batch, captions)
        return batch

    @staticmethod
    def _gather(found: list, keeps: list) -> torch.Tensor:
        """Concatenate full-grid latents, keeping only `keeps[i]` rows of sample i."""
        return torch.cat(
            [t if k is None else t.index_select(0, k) for t, k in zip(found, keeps)]
        )

    def _latent_lookup(self, batch: HostBatch, items) -> list:
        keys = [self.latents.key(self.catalog[it.row_id], it) for it in items]
        batch.latent_keys = keys
        return [self.latents.get(k) for k in keys]

    def _latents_decode(self, batch: HostBatch, items) -> None:
        batch.pixels = [self.pixels(it) for it in items]

    def _latents_offline(self, batch: HostBatch, items) -> None:
        found = self._latent_lookup(batch, items)
        if any(t is None for t in found):
            it = items[_first_missing(found)]
            row = self.catalog[it.row_id]
            first = (
                f"row {it.row_id} {row.locator} {row.w}x{row.h} -> target "
                f"{it.target_hw[1]}x{it.target_hw[0]} crop {it.crop_ltrb} flip {it.flip}"
            )
            n = sum(t is None for t in found)
            raise offline_miss(self.cfg, "latents", n, first)
        batch.latents = self._gather(found, batch.img_keep)

    def _latents_online(self, batch: HostBatch, items) -> None:
        found = self._latent_lookup(batch, items)
        if any(t is None for t in found):
            self._latents_decode(batch, items)
            return
        batch.latents = self._gather(found, batch.img_keep)

    def _text_lookup(self, batch: HostBatch, captions: list[str]) -> list:
        keys = [self.texts.key(c) for c in captions]
        batch.text_keys = keys
        return [self.texts.get(k) for k in keys]

    def _text_prompts(self, batch: HostBatch, captions: list[str]) -> None:
        batch.prompts = captions

    @staticmethod
    def _set_text(batch: HostBatch, found: list) -> None:
        cut = [t[: limit or None] for t, limit in zip(found, batch.text_limit)]
        batch.text = torch.cat(cut)
        batch.text_len = torch.tensor([t.shape[0] for t in cut], dtype=torch.int32)

    def _text_offline(self, batch: HostBatch, captions: list[str]) -> None:
        found = self._text_lookup(batch, captions)
        if any(t is None for t in found):
            caption = captions[_first_missing(found)]
            first = f"caption {caption[:80]!r}"
            n = sum(t is None for t in found)
            raise offline_miss(self.cfg, "text", n, first)
        self._set_text(batch, found)

    def _text_online(self, batch: HostBatch, captions: list[str]) -> None:
        found = self._text_lookup(batch, captions)
        if any(t is None for t in found):
            batch.prompts = captions
            return
        self._set_text(batch, found)

    def _write_latents(self, batch: HostBatch, latents: torch.Tensor) -> None:
        sizes = (batch.hw[:, 0].long() * batch.hw[:, 1].long()).tolist()
        for key, part in zip(batch.latent_keys, latents.split(sizes)):
            self.latents.put(key, part)

    def _write_text(self, batch: HostBatch, text: list[torch.Tensor]) -> None:
        for key, emb in zip(batch.text_keys, text):
            self.texts.put(key, emb)

    def write_back(
        self,
        batch: HostBatch,
        latents: torch.Tensor | None = None,
        text: list[torch.Tensor] | None = None,
    ) -> None:
        """Online caches: store the trainer's encodings of a batch under its keys.
        `latents` is packed [N_img, C] in batch order; `text` is one tensor per sample;
        a cache that is not online ignores its argument."""
        if latents is not None:
            self._put_latents(self, batch, latents)
        if text is not None:
            self._put_text(self, batch, text)
