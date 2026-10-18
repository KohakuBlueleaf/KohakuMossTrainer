"""Aspect-ratio buckets: every batch holds samples of one target size."""

from kohakumoss_data.registry import PLANNERS
from kohakumoss_data.resolution.base import Planner
from kohakumoss_data.resolution.geometry import nearest_bucket


@PLANNERS.register("bucket")
class BucketPlanner(Planner):
    """Rows go to the nearest log-aspect bucket of ~area; with `upscale="never"` a row
    smaller than its bucket gets its own native-size bucket instead. Batching runs per
    bucket (a batch never mixes sizes); batches are then shuffled."""

    grouped = True

    def __init__(self, catalog, geometry, **kwargs):
        self.bucket_list = geometry.buckets()
        super().__init__(catalog, geometry, **kwargs)

    def target(self, row_id: int) -> tuple[int, int]:
        w, h = int(self.catalog.w[row_id]), int(self.catalog.h[row_id])
        bucket = nearest_bucket(w, h, self.bucket_list)
        if self.geometry.upscale == "never" and w * h < bucket[0] * bucket[1]:
            return self.geometry.fit(w, h)
        return bucket

    def _streams(self, items: list, tokens: list[int]) -> list[tuple[list, list]]:
        groups: dict[tuple[int, int], tuple[list, list]] = {}
        for item, t in zip(items, tokens):
            group = groups.setdefault(item.target_hw, ([], []))
            group[0].append(item)
            group[1].append(t)
        return [groups[k] for k in sorted(groups)]
