"""Loss scalers. The scale is a host float identical on every rank (DeviceScaler: a
device tensor, for the deferred step); the skip decision comes from one device-side
finite flag agreed across ranks (see OptimBundle.step)."""

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

    def update_device(self, bad: Tensor) -> None:
        return None

    def host_scale(self) -> float:
        return 1.0

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

    def host_scale(self) -> float:
        return float(self.scale)

    def state_dict(self) -> dict:
        return {"scale": self.scale, "clean_steps": self.clean_steps}

    def load_state_dict(self, state: dict) -> None:
        self.scale = float(state["scale"])
        self.clean_steps = int(state["clean_steps"])


class DeviceScaler(DynamicScaler):
    """DynamicScaler whose scale and clean-step count are 0-d fp32 tensors on `device`,
    updated from the on-device skip flag (`update_device`, no host read); the same
    sequence of scales as DynamicScaler.update. `host_scale()` reads the scale."""

    def __init__(self, base: DynamicScaler, device: torch.device):
        super().__init__(
            base.scale, base.growth_interval, base.growth, base.backoff, base.floor
        )
        self.scale = torch.tensor(float(base.scale), device=device)
        self.clean = torch.tensor(float(base.clean_steps), device=device)

    def update(self, found_inf: bool) -> None:
        raise RuntimeError("DeviceScaler updates on the device: update_device(flag)")

    def update_device(self, bad: Tensor) -> None:
        """One step with `bad` (0-d fp32, 1 = non-finite): back off (floored) and reset
        the clean count, else count it and grow after `growth_interval` clean steps."""
        clean = (self.clean + 1.0) * (1.0 - bad)
        grow = (clean >= self.growth_interval).float()
        grown = self.scale * (1.0 + (self.growth - 1.0) * grow)
        backed = (self.scale * self.backoff).clamp(min=self.floor)
        self.scale = torch.where(bad > 0, backed, grown)
        self.clean = clean * (1.0 - grow)

    def host_scale(self) -> float:
        return float(self.scale)

    def state_dict(self) -> dict:
        return {"scale": float(self.scale), "clean_steps": int(self.clean)}

    def load_state_dict(self, state: dict) -> None:
        self.scale.fill_(float(state["scale"]))
        self.clean.fill_(float(state["clean_steps"]))
