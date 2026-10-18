"""The MOSS inference schedule: shifted sigmas snapped to the discrete training grid."""

import torch
from torch import Tensor

from kohakumoss.diffusion.timestep.shift import shift_sigma


def _grid(count: int, shift: float, total: int) -> Tensor:
    lower = shift / (total + shift - 1)
    return shift_sigma(torch.linspace(1, lower, count, dtype=torch.float32), shift)


def moss_sigmas(
    steps: int = 50, shift: float = 5.0, total: int = 1000, training_shift: float = 1.0
) -> Tensor:
    """Descending fp32 sigmas ending with 0: `steps` shifted values, each snapped to the
    nearest training sigma, consecutive duplicates dropped."""
    if steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError("steps and shifts must be positive")
    sigmas = _grid(steps, shift, total)
    training = _grid(total, training_shift, total).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearer_lower = (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs()
    snapped = torch.unique_consecutive(
        training[torch.where(nearer_lower, lower, upper)]
    )
    return torch.cat((snapped, snapped.new_zeros(1)))
