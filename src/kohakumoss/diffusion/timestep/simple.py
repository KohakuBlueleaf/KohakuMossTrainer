"""Continuous samplers: uniform and logit-normal (sigma = sigmoid(mu + s Z)), with shift.

The shift is bound at construction (identity when shift == 1).
"""

from functools import partial

import torch
from torch import Tensor

from kohakumoss.diffusion.timestep.shift import shift_sigma
from kohakumoss_common.calls import identity


def bind_shift(shift: float):
    """`sigma -> shifted sigma`, the identity for shift 1."""
    return identity if shift == 1.0 else partial(shift_sigma, shift=shift)


class UniformTimestep:
    """sigma ~ U(min, max), then shifted."""

    def __init__(self, shift: float = 1.0, low: float = 0.001, high: float = 1.0):
        self.shift = bind_shift(shift)
        self.low = low
        self.span = high - low

    def sample(self, batch: int, generator: torch.Generator, step: int) -> Tensor:
        u = torch.rand(batch, generator=generator)
        return self.shift(self.low + self.span * u)


class LogitNormalTimestep:
    """sigma = sigmoid(mu + s * Z), then shifted (SD3 convention: larger mu = noisier)."""

    def __init__(self, mu: float = 0.0, s: float = 1.0, shift: float = 1.0):
        self.mu = mu
        self.s = s
        self.shift = bind_shift(shift)

    def sample(self, batch: int, generator: torch.Generator, step: int) -> Tensor:
        z = torch.randn(batch, generator=generator)
        return self.shift(torch.sigmoid(self.mu + self.s * z))
