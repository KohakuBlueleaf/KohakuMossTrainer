"""Sampler schedule, model-free: equals the official grid snap. Euler + CFG on the real
model runs in tests/integration/test_infer_real.py."""

import pytest
import torch

from kohakumoss.diffusion.sampling import moss_sigmas

pytestmark = pytest.mark.cpu


def _official_sigmas(steps, shift, total=1000, training_shift=1.0):
    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where(
        (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(),
        lower,
        upper,
    )
    return torch.unique_consecutive(training[nearest])


@pytest.mark.parametrize("steps, shift", [(50, 5.0), (28, 3.0), (8, 1.0)])
def test_schedule_matches_official(steps, shift):
    ours = moss_sigmas(steps, shift)
    ref = _official_sigmas(steps, shift)
    assert ours[-1] == 0 and torch.equal(ours[:-1], ref)
    grid = torch.arange(1, 1001) / 1000
    assert all(((grid - s).abs().min() < 1e-6) for s in ref)
