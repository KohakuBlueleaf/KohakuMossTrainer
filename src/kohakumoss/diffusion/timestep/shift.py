"""Static timestep shift `s*t / (1 + (s-1)*t)`."""

import torch
from torch import Tensor


def shift_sigma(sigma: Tensor, shift: float) -> Tensor:
    """Shift sigma in [0, 1] towards noise for shift > 1 (identity at shift == 1)."""
    if shift == 1.0:
        return sigma
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def shift_table(table: Tensor, shift: float) -> Tensor:
    """`shift_sigma` applied to a precomputed table, in float32."""
    return shift_sigma(table.to(torch.float32), shift)
