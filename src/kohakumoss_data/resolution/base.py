"""Planner base: target size per row x batching function x over-long fit; epoch `e` is a
pure function of the seed and the catalog, global batch `g = index * world + rank`.
See docs/concepts/data.md#resolution-batching-over-long-samples."""

from collections.abc import Callable

import numpy as np

from kohakumoss_data.format.catalog import Catalog
from kohakumoss_data.resolution.batching import BATCHINGS, BatchLimits
from kohakumoss_data.resolution.geometry import Geometry
from kohakumoss_data.resolution.overlong import OVERLONG
from kohakumoss_data.seeds import rng_of, seed_of
from kohakumoss_data.types import BatchPlan, PlanItem

TextLen = Callable[[int, int], int]
EPOCH_CACHE = 5


def split_weights(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float32 weights [N] -> (whole repeats int64 [N], fractional part float32 [N])."""
    weights = np.asarray(weights, dtype=np.float32)
    floor = np.floor(weights)
    return floor.astype(np.int64), weights - floor


def expected_items(weights: np.ndarray) -> float:
    """Expected epoch size: exact int64 sum of the whole repeats plus the pairwise
    float32 sum of the fractional parts."""
    whole, frac = split_weights(weights)
    return int(whole.sum()) + float(np.sum(frac, dtype=np.float32))


def epoch_order(weights: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    """Row ids of epoch `epoch`: row i appears floor(w_i) times plus once more with
    probability frac(w_i), then the multiset is permuted."""
    rng = rng_of("epoch-order", seed, epoch)
    whole, frac = split_weights(weights)
    extra = rng.random(len(frac), dtype=np.float32) < frac
    counts = whole + extra
    items = np.repeat(np.arange(len(weights), dtype=np.int64), counts)
    return items[rng.permutation(len(items))]


def no_text(row_id: int, variant: int) -> int:
    return 0


def draw_flip(rng: np.random.Generator) -> bool:
    return bool(rng.random() < 0.5)


def no_flip(rng: np.random.Generator) -> bool:
    return False


def in_order(batches: list, seed: int, epoch: int) -> list:
    return batches


def shuffled(batches: list, seed: int, epoch: int) -> list:
    rng = rng_of("batch-order", seed, epoch)
    return [batches[i] for i in rng.permutation(len(batches))]


class Planner:
    """Shared machinery. `batching`: tokens | tokens_ffd | count; `overlong`: drop |
    truncate_text | drop_tokens. Subclasses implement `target(row_id)` and may split
    the stream into groups (`_streams`)."""

    grouped = False

    def __init__(
        self,
        catalog: Catalog,
        geometry: Geometry,
        seed: int = 0,
        flip: bool = False,
        caption_variants: int = 1,
        text_len: TextLen | None = None,
        batching: str = "tokens",
        overlong: str = "drop",
        max_tokens: int = 16384,
        batch_size: int = 4,
        window: int = 256,
        max_samples: int = 0,
        drop_last: bool = True,
    ):
        if batching not in BATCHINGS:
            raise ValueError(f"batching must be one of {sorted(BATCHINGS)}")
        if overlong not in OVERLONG:
            raise ValueError(f"overlong must be one of {sorted(OVERLONG)}")
        self.catalog = catalog
        self.geometry = geometry
        self.seed = seed
        self.caption_variants = max(1, caption_variants)
        self.text_len = text_len or no_text
        self.limits = BatchLimits(
            max_tokens, batch_size, window, max_samples, drop_last
        )
        self._box = geometry.box_fn()
        self._flip = draw_flip if flip else no_flip
        self._batch = BATCHINGS[batching]
        self._overlong = OVERLONG[overlong]
        self._arrange = (
            shuffled if self.grouped or batching == "tokens_ffd" else in_order
        )
        self.targets = [self.target(i) for i in range(len(catalog))]
        self._epochs: dict[int, list[list[PlanItem]]] = {}
        self._counts: list[int] = []
        self.drops: dict[int, int] = {}
        self.unbatched: dict[int, int] = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_epochs"] = {}
        state["_counts"] = []
        return state

    def target(self, row_id: int) -> tuple[int, int]:
        """Target (h, w) in pixels of row `row_id`."""
        raise NotImplementedError

    def item(self, row_id: int, epoch: int, position: int, target) -> PlanItem:
        """PlanItem for one occurrence: crop, flip and caption variant are seeded."""
        w, h = int(self.catalog.w[row_id]), int(self.catalog.h[row_id])
        rng = rng_of("item", self.seed, epoch, position, row_id)
        box = self._box(w, h, target, rng)
        flip = self._flip(rng)
        variant = int(rng.integers(self.caption_variants))
        return PlanItem(row_id, tuple(target), box, flip, variant)

    def sizes(self, item: PlanItem) -> tuple[int, int]:
        """(image tokens, text tokens) of `item` after its over-long fit."""
        img = item.img_keep or self.geometry.tokens(item.target_hw)
        text = self.text_len(item.row_id, item.caption_variant)
        return img, min(text, item.text_limit or text)

    def tokens(self, item: PlanItem) -> int:
        """Joint token count of `item`."""
        img, text = self.sizes(item)
        return img + text

    def _streams(self, items: list, tokens: list[int]) -> list[tuple[list, list]]:
        return [(items, tokens)]

    def _epoch(self, e: int) -> list[list[PlanItem]]:
        limit = self.limits.max_tokens
        items, tokens, dropped = [], [], 0
        for pos, row_id in enumerate(epoch_order(self.catalog.weight, self.seed, e)):
            item = self.item(int(row_id), e, pos, self.targets[row_id])
            img, text = self.sizes(item)
            if img + text > limit:
                keep_seed = seed_of("img-keep", self.seed, e, pos, int(row_id))
                item = self._overlong(item, img, text, limit, keep_seed)
                if item is None:
                    dropped += 1
                    continue
                img, text = self.sizes(item)
            items.append(item)
            tokens.append(img + text)
        batches = [
            batch
            for stream in self._streams(items, tokens)
            for batch in self._batch(*stream, self.limits)
        ]
        self.drops[e] = dropped
        self.unbatched[e] = len(items) - sum(len(b) for b in batches)
        return self._arrange(batches, self.seed, e)

    def epoch(self, e: int) -> list[list[PlanItem]]:
        """The batches of epoch `e` (the last EPOCH_CACHE epochs are kept)."""
        if e not in self._epochs:
            if len(self._epochs) >= EPOCH_CACHE:
                self._epochs.pop(min(self._epochs))
            self._epochs[e] = self._epoch(e)
        return self._epochs[e]

    def _count(self, e: int) -> int:
        while len(self._counts) <= e:
            self._counts.append(len(self.epoch(len(self._counts))))
        return self._counts[e]

    def locate(self, g: int) -> tuple[int, int]:
        """(epoch, local index) of global batch `g`."""
        e = 0
        while True:
            n = self._count(e)
            if n == 0:
                raise ValueError("planner produced an empty epoch")
            if g < n:
                return e, g
            g -= n
            e += 1

    def plan(self, index: int, rank: int, world: int) -> BatchPlan:
        """Batch `index` of `rank` among `world` ranks."""
        e, local = self.locate(index * world + rank)
        items = self.epoch(e)[local]
        budget = sum(self.tokens(it) for it in items)
        return BatchPlan(index=index, items=list(items), token_budget=budget)

    def epoch_length(self, e: int = 0) -> int:
        """Batches in epoch `e`."""
        return self._count(e)
