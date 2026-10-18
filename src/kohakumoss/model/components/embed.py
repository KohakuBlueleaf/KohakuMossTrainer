"""Timestep sinusoid, time-embedding MLP and RoPE frequency bands."""

import torch
from torch import Tensor, nn

from kohakumoss.model.linear import MossLinear


def timestep_embedding(t: Tensor, freq_dim: int) -> Tensor:
    """t [B] (0..1000) -> fp32 [B, freq_dim] = [cos | sin] of t * 10000^(-k/half): the
    official MOSS formula, in fp32. See docs/concepts/numerics.md."""
    half = freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=t.device, dtype=torch.float32) / half)
    phase = t.float()[:, None] * freq
    return torch.cat((phase.cos(), phase.sin()), -1)


class TimeEmbedding(nn.Sequential):
    """Linear(freq_dim, dim) -> SiLU -> Linear(dim, dim); keys `0.*` and `2.*`."""

    def __init__(self, freq_dim: int, dim: int, device=None, dtype=None):
        super().__init__(
            MossLinear(freq_dim, dim, device=device, dtype=dtype),
            nn.SiLU(),
            MossLinear(dim, dim, device=device, dtype=dtype),
        )


def default_bands(n_bands: int, base: float, device=None) -> Tensor:
    """1 / base^(j / n_bands), built in fp32 on CPU, then moved to `device`."""
    exponent = torch.arange(n_bands, dtype=torch.float32) / n_bands
    return (1 / (base**exponent)).to(device)


class RopeBands(nn.Module):
    """Holds the fp32 `bands` buffer; any dtype cast of the module rebuilds it in fp32."""

    def __init__(self, n_bands: int, base: float, device=None):
        super().__init__()
        self.n_bands = n_bands
        self.base = base
        device = None if str(device) == "meta" else device
        self.register_buffer("bands", default_bands(n_bands, base, device))

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        device = self.bands.device
        device = None if device.type == "meta" else device
        self.bands = default_bands(self.n_bands, self.base, device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.bands = default_bands(self.n_bands, self.base, self.bands.device)
