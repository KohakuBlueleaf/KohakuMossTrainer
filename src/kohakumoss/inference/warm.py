"""Kernel prewarm for inference: tune the model's ops for the lengths a call will hit."""

import torch

from kohakumoss.training.build.kernels import Prewarm
from kohakumoss_kernels.backends.compile.race import retune_pending


class _NoWarm:
    """CPU, shared (training-owned) models and TUNE='off': nothing is timed."""

    def ensure(self, sample_max: int, batch_max: int) -> None:
        return None

    def retune(self) -> None:
        return None


no_warm = _NoWarm()


class KernelWarm:
    """Prewarms (outside inference mode) once per larger (sample, batch) envelope and
    decides problems a call queued afterwards."""

    def __init__(self, backbone, device: torch.device):
        self.backbone = backbone
        self.device = device
        self.sample = 0
        self.batch = 0
        self.measured = 0

    def ensure(self, sample_max: int, batch_max: int) -> None:
        if sample_max <= self.sample and batch_max <= self.batch:
            return
        self.sample = max(self.sample, sample_max)
        self.batch = max(self.batch, batch_max)
        dtype = self.backbone.layers[0].proj["q"].weight.dtype
        warm = Prewarm(
            self.backbone.ops, self.backbone.cfg, dtype, self.sample, self.batch
        )
        self.measured += warm([self.device])

    def retune(self) -> None:
        self.measured += retune_pending()
