"""Kernel prewarm for inference: tune the model's ops, forward only and under no_grad,
for the lengths a call will hit (the record / replay prewarm; problems already decided
in the tables are not run); the tuning workspaces are returned to the device after it.
"""

from collections.abc import Callable

import torch

from kohakumoss.logging.log import get_logger, timed
from kohakumoss.model.backbone import compute_dtype
from kohakumoss.offload.accel import release_cache
from kohakumoss.precision import autocast_for
from kohakumoss.tuning.prewarm import ModelPrewarm
from kohakumoss_kernels.backends.compile.choice import retune_pending
from kohakumoss_kernels.tuning import decisions

log = get_logger(__name__)


class _NoWarm:
    """CPU, shared (training-owned) models and TUNE='off': nothing is timed."""

    def ensure(self, sample_max: int, batch_max: int) -> None:
        return None

    def retune(self) -> None:
        return None


no_warm = _NoWarm()


def prewarm_autocast(backbone, device) -> Callable:
    """The context factory the inference prewarm records under: the sampler's
    (`inference.pipeline.sampling_autocast`), so its problem keys carry the dtypes a
    sampling call meets."""
    return autocast_for(compute_dtype(backbone), device)


__all__ = ["KernelWarm", "no_warm", "release_cache"]


class KernelWarm:
    """Prewarms (forward only, no autograd) once per larger (sample, batch) envelope and
    decides problems a call queued afterwards."""

    def __init__(self, backbone, device: torch.device):
        self.backbone = backbone
        self.device = device
        self.sample = 0
        self.batch = 0
        self.measured = 0

    @torch.no_grad()
    def ensure(self, sample_max: int, batch_max: int) -> None:
        if sample_max <= self.sample and batch_max <= self.batch:
            return
        self.sample = max(self.sample, sample_max)
        self.batch = max(self.batch, batch_max)
        dtype = compute_dtype(self.backbone)
        warm = ModelPrewarm(
            self.backbone,
            None,
            prewarm_autocast(self.backbone, self.device),
            dtype,
            self.sample,
            self.batch,
            backward=False,
        )
        what = (
            f"kernel prewarm on {self.device} (sample {self.sample} / batch "
            f"{self.batch} tokens; tuned results are cached for later runs)"
        )
        with timed(log, what) as done:
            measured = warm([self.device])
            done["problems"] = measured
        decisions.flush("after the inference prewarm")
        self.measured += measured
        release_cache(self.device)

    @torch.no_grad()
    def retune(self) -> None:
        self.measured += retune_pending()
        release_cache(self.device)
