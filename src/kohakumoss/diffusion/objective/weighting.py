"""Per-sample loss weights as a function of sigma, multiplied into the data weight."""

from torch import Tensor


def no_weighting(sigma: Tensor, weight: Tensor) -> Tensor:
    """Velocity MSE unweighted (the MOSS recipe)."""
    return weight


def x0_weighting(sigma: Tensor, weight: Tensor) -> Tensor:
    """sigma^2: velocity MSE equal to x0-space MSE."""
    return weight * sigma.to(weight.dtype).square()


def eps_weighting(sigma: Tensor, weight: Tensor) -> Tensor:
    """(1 - sigma)^2: velocity MSE equal to noise-space MSE."""
    return weight * (1.0 - sigma.to(weight.dtype)).square()
