"""Timestep (sigma) samplers, selected through TIMESTEPS."""

from kohakumoss.diffusion.timestep.moss import (
    TECH_REPORT_STAGE3,
    MossTimestep,
    moss_pmf,
)
from kohakumoss.diffusion.timestep.shift import shift_sigma
from kohakumoss.diffusion.timestep.simple import LogitNormalTimestep, UniformTimestep
from kohakumoss_common.registry import Registry

TIMESTEPS = Registry("timestep sampler")
TIMESTEPS.add("moss", MossTimestep)
TIMESTEPS.add("uniform", UniformTimestep)
TIMESTEPS.add("logit_normal", LogitNormalTimestep)

__all__ = [
    "TECH_REPORT_STAGE3",
    "TIMESTEPS",
    "LogitNormalTimestep",
    "MossTimestep",
    "UniformTimestep",
    "moss_pmf",
    "shift_sigma",
]
