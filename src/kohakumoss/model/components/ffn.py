"""SiTU on separate up / gate tensors (the text refiner's two linears), no concatenation."""

import torch
from torch import Tensor


def situ_pair(u: Tensor, g: Tensor) -> Tensor:
    """(4 tanh(g/4) sigmoid(g)) * (25 tanh(u/25)) in >= fp32, rounded once to u's dtype;
    equals `situ_glu(cat(u, g))`."""
    dt = torch.promote_types(u.dtype, torch.float32)
    uf, gf = u.to(dt), g.to(dt)
    a = (4.0 * torch.tanh(gf / 4.0) * torch.sigmoid(gf)) * (
        25.0 * torch.tanh(uf / 25.0)
    )
    return a.to(u.dtype)
