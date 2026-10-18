"""Loss scalers. The scale is a host float identical on every rank; the skip decision
comes from one device-side finite flag agreed across ranks (see OptimBundle.step)."""

import torch
from torch import Tensor


class NoScaler:
    """bf16 / fp32: scale 1, nothing to unscale."""

    scale = 1.0

    def scale_loss(self, loss: Tensor) -> Tensor:
        return loss

    def unscale_(self, grads: list[Tensor]) -> None:
        return None

    def update(self, found_inf: bool) -> None:
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None


class DynamicScaler:
    """fp16 dynamic loss scale: backoff on overflow (floored), growth after a clean run."""

    def __init__(
        self,
        init: float = 2.0**16,
        growth_interval: int = 2000,
        growth: float = 2.0,
        backoff: float = 0.5,
        floor: float = 1.0,
    ):
        self.scale = float(init)
        self.growth_interval = growth_interval
        self.growth = growth
        self.backoff = backoff
        self.floor = floor
        self.clean_steps = 0

    def scale_loss(self, loss: Tensor) -> Tensor:
        return loss * self.scale

    def unscale_(self, grads: list[Tensor]) -> None:
        if grads:
            torch._foreach_mul_(grads, 1.0 / self.scale)

    def update(self, found_inf: bool) -> None:
        if found_inf:
            self.scale = max(self.floor, self.scale * self.backoff)
            self.clean_steps = 0
            return
        self.clean_steps += 1
        if self.clean_steps >= self.growth_interval:
            self.scale *= self.growth
            self.clean_steps = 0

    def state_dict(self) -> dict:
        return {"scale": self.scale, "clean_steps": self.clean_steps}

    def load_state_dict(self, state: dict) -> None:
        self.scale = float(state["scale"])
        self.clean_steps = int(state["clean_steps"])
