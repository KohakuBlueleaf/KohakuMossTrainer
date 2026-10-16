"""Square resolutions: centre crop (`square_crop`) or seeded-shift crop (`shifted_square`)."""

from dataclasses import replace

from kohakumoss_data.registry import PLANNERS
from kohakumoss_data.resolution.base import Planner


@PLANNERS.register("square_crop")
class SquareCropPlanner(Planner):
    """sqrt(area)-sided square crops (always resized to that side; `upscale` does not
    apply)."""

    crop_mode = "center"

    def __init__(self, catalog, geometry, **kwargs):
        super().__init__(catalog, replace(geometry, crop=self.crop_mode), **kwargs)

    def target(self, row_id: int) -> tuple[int, int]:
        side = self.geometry.square()
        return side, side


@PLANNERS.register("shifted_square")
class ShiftedSquarePlanner(SquareCropPlanner):
    """Square crops at a seeded random position."""

    crop_mode = "random"
