"""`resize`: native aspect, resized so the area is ~TARGET_AREA (sides multiples of step)."""

from kohakumoss_data.registry import PLANNERS
from kohakumoss_data.resolution.base import Planner


@PLANNERS.register("resize")
class ResizePlanner(Planner):
    """Each row keeps its aspect (clamped to `aspect_range`); every sample has nearly the
    same token count. `upscale="never"` keeps smaller images at their native size."""

    def target(self, row_id: int) -> tuple[int, int]:
        w, h = int(self.catalog.w[row_id]), int(self.catalog.h[row_id])
        return self.geometry.fit(w, h)
