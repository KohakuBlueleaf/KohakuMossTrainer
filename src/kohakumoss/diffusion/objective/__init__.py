"""Training objective: flow noising, token losses, sigma weighting, normalisation."""

from kohakumoss.diffusion.objective.flow import noise_latents
from kohakumoss.diffusion.objective.loss import HuberLoss, L1Loss, MSELoss
from kohakumoss.diffusion.objective.normalize import TokenNormalizer, all_reduce_sum
from kohakumoss.diffusion.objective.weighting import (
    eps_weighting,
    no_weighting,
    x0_weighting,
)
from kohakumoss_common.registry import Registry

LOSSES = Registry("loss")
LOSSES.add("mse", MSELoss)
LOSSES.add("masked_mse", MSELoss)
LOSSES.add("huber", HuberLoss)
LOSSES.add("l1", L1Loss)

WEIGHTINGS = Registry("loss weighting")
WEIGHTINGS.add("none", lambda: no_weighting)
WEIGHTINGS.add("x0", lambda: x0_weighting)
WEIGHTINGS.add("eps", lambda: eps_weighting)

__all__ = [
    "LOSSES",
    "WEIGHTINGS",
    "HuberLoss",
    "L1Loss",
    "MSELoss",
    "TokenNormalizer",
    "all_reduce_sum",
    "noise_latents",
]
