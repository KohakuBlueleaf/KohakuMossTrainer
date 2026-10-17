"""Timestep sinusoid, time-embedding MLP and RoPE frequency bands."""

import math

import torch
from torch import Tensor, nn

from kohakumoss.model.linear import MossLinear

HI_BITS = 10
T_BITS = 14


def _split(x: float) -> tuple[float, float]:
    """x = hi + lo with hi carrying HI_BITS significant bits (host constants)."""
    if x == 0.0:
        return 0.0, 0.0
    m, e = math.frexp(x)
    hi = math.ldexp(round(m * 2**HI_BITS), e - HI_BITS)
    return hi, x - hi


class Sinusoid:
    """t [B] (0 <= t <= t_max) -> fp32 [B, freq_dim] = [cos | sin] of t * 10000^(-k/half),
    in fp32 with the phase reduced in turns. See docs/concepts/numerics.md."""

    def __init__(self, freq_dim: int, t_max: float):
        half = freq_dim // 2
        turns = [10000 ** (-k / half) / (2 * math.pi) for k in range(half)]
        self.parts = tuple(zip(*(_split(g) for g in turns)))
        self.turns = turns
        # t_hi keeps T_BITS significant bits for t <= t_max
        self.t_scale = 2.0 ** (T_BITS - max(1, math.ceil(math.log2(t_max + 1))))
        self._tables: dict[torch.device, tuple[Tensor, Tensor, Tensor]] = {}

    def _table(self, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        if device not in self._tables:
            hi, lo = self.parts
            with torch.inference_mode(False):
                self._tables[device] = tuple(
                    torch.tensor(v, dtype=torch.float32, device=device)
                    for v in (hi, lo, self.turns)
                )
        return self._tables[device]

    def __call__(self, t: Tensor) -> Tensor:
        g_hi, g_lo, g = self._table(t.device)
        t = t.float()[:, None]
        t_hi = torch.floor(t * self.t_scale) / self.t_scale
        t_lo = t - t_hi
        exact = t_hi * g_hi
        whole = torch.round(exact)
        frac = (exact - whole) + (t_hi * g_lo + t_lo * g)
        phase = frac * (2 * math.pi)
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
