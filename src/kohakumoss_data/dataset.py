"""MossDataset: sources + catalog + captions + planner + caches -> HostBatch per index.

Everything fixed by the config (cache modes, text-length rule, text-dropout rule,
decoder, write-back) is bound at construction; `materialize` only calls the bound
functions. What still branches per batch is data: the seeded dropout draw and, in
online mode, whether the cache actually holds a key.
"""

import math
import os
from functools import partial

import numpy as np
import torch

from kohakumoss_data.cache.store import LengthTable
from kohakumoss_data.cache.tensor_cache import LatentCache, TextCache
from kohakumoss_data.caption.template import build_template
from kohakumoss_data.config import DataConfig, resolved_cache_dir
from kohakumoss_data.format.catalog import Catalog, read_catalog
from kohakumoss_data.loader.decode import Decoder
from kohakumoss_data.registry import PLANNERS, SOURCES, build
from kohakumoss_data.resolution.geometry import Geometry
from kohakumoss_data.resolution.item import keep_indices
from kohakumoss_data.seeds import rng_of
from kohakumoss_data.types import HostBatch, PlanItem

DATASET_KEYS = ("weight", "caption", "loss_weight", "catalog")


def build_source(spec: dict):
    """A source from a dataset entry (the dataset-level keys are stripped)."""
    kwargs = {k: v for k, v in spec.items() if k not in DATASET_KEYS}
    return build(kwargs, SOURCES, key="source")


def scan_source(spec: dict, source) -> Catalog:
    """Rows of one dataset entry, from its prebuilt `catalog` file when given."""
    path = spec.get("catalog")
    if path and os.path.exists(path):
        return read_catalog(path)
    return Catalog(list(source.scan()))


def draw_drop(seed: int, p: float, index: int, rank: int, slot: int) -> bool:
    """Seeded text-dropout draw for one sample slot."""
    return bool(rng_of("text-drop", seed, index, rank, slot).random() < p)


def never_drop(index: int, rank: int, slot: int) -> bool:
    return False


def _ignore(batch, value) -> None:
    return None


def _missing(what: str, script: str, found: list) -> KeyError:
    n = sum(t is None for t in found)
    return KeyError(
        f"{n} {what} missing from the offline cache; run {script} "
        "with the same data config"
    )


class MossDataset:
    """Pure function `materialize(index, rank, world) -> HostBatch`; picklable."""

    def __init__(self, cfg: DataConfig):
        if not cfg.datasets:
            raise ValueError("DATASETS is empty")
        self.cfg = cfg
        self.sources = [build_source(spec) for spec in cfg.datasets]
        parts = [
            scan_source(spec, src) for spec, src in zip(cfg.datasets, self.sources)
        ]
        weights = [float(spec.get("weight", 1.0)) for spec in cfg.datasets]
        self.catalog = Catalog.concat(parts, weights)
        self.templates = [
            build_template(spec.get("caption", cfg.caption)) for spec in cfg.datasets
        ]
        self.loss_weight = np.array(
            [float(spec.get("loss_weight", 1.0)) for spec in cfg.datasets]
        )
        cache_dir = resolved_cache_dir(cfg)
        self.latents = LatentCache(
            os.path.join(cache_dir, "latents.db") if cache_dir else None,
            cfg.latent_cache,
            cfg.latent_encoder_id,
            cfg.cache_dtype,
            str(cfg.background),
            cfg.resample,
            cfg.latent_draw,
        )
        self.texts = TextCache(
            os.path.join(cache_dir, "text.db") if cache_dir else None,
            cfg.text_cache,
            cfg.text_encoder_id,
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
        offline = cfg.text_cache == "offline"
        self._text_lengths = self.texts.lengths() if offline else LengthTable()
        self._bind()
        self.planner = self._planner()

    def _bind(self) -> None:
        cfg = self.cfg
        if cfg.token_budget not in ("estimate", "exact"):
            raise ValueError("token_budget must be 'estimate' or 'exact'")
        if cfg.token_budget == "exact" and cfg.text_cache != "offline":
            raise ValueError("token_budget='exact' needs text_cache='offline'")
        self.text_len = {
            ("estimate", "off"): self._text_len_estimate,
            ("estimate", "online"): self._text_len_estimate,
            ("estimate", "offline"): self._text_len_cached,
            ("exact", "offline"): self._text_len_exact,
        }[(cfg.token_budget, cfg.text_cache)]
        self._drop = (
            partial(draw_drop, cfg.seed, cfg.text_drop) if cfg.text_drop else never_drop
        )
        self._fill_latents = {
            "off": self._latents_decode,
            "offline": self._latents_offline,
            "online": self._latents_online,
        }[cfg.latent_cache]
        self._fill_text = {
            "off": self._text_prompts,
            "offline": self._text_offline,
            "online": self._text_online,
        }[cfg.text_cache]
        latent_online = cfg.latent_cache == "online"
        text_online = cfg.text_cache == "online"
        self._put_latents = self._write_latents if latent_online else _ignore
        self._put_text = self._write_text if text_online else _ignore

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
        template = self.templates[int(self.catalog.source[row_id])]
        return template.render(row, rng_of("caption", self.cfg.seed, row_id, variant))

    def estimate_text_tokens(self, caption: str) -> int:
        n = math.ceil(len(caption) / self.cfg.text_chars_per_token)
        return min(n + self.cfg.text_overhead_tokens, self.cfg.max_text_tokens)

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
        source = self.sources[int(self.catalog.source[row_id])]
        return source.open(self.catalog[row_id].locator)

    def pixels(self, item: PlanItem) -> np.ndarray:
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
        self._fill_latents(batch, items)
        self._fill_text(batch, captions)
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
            raise _missing("latents", "scripts/data/cache_latents.py", found)
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
            raise _missing("text embeddings", "scripts/data/cache_text.py", found)
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
            self._put_latents(batch, latents)
        if text is not None:
            self._put_text(batch, text)
