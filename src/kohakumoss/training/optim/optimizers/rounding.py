"""Writing fp32 updates back into 16-bit parameters: nearest, stochastic, Kahan."""

import torch
from torch import Tensor


def write_nearest(p: Tensor, new: Tensor, comp: Tensor | None) -> None:
    p.copy_(new)


def write_stochastic(p: Tensor, new: Tensor, comp: Tensor | None) -> None:
    """fp32 -> bf16 with probability proportional to distance (unbiased)."""
    bits = new.contiguous().view(torch.int32)
    noise = torch.randint_like(bits, 0, 1 << 16)
    rounded = ((bits + noise) & -65536).view(torch.float32)
    p.copy_(rounded)


def write_kahan(p: Tensor, new: Tensor, comp: Tensor | None) -> None:
    """Round to nearest and keep the rounding error in `comp` for the next update."""
    target = new + comp.float()
    p.copy_(target)
    comp.copy_(target - p.float())


WRITERS = {
    "nearest": write_nearest,
    "stochastic": write_stochastic,
    "kahan": write_kahan,
}
