"""Channel-weighted RMSNorm ops and the heal anneal.

`WeightedNorms` normalises by sqrt(mean(lam * x^2) + eps), lam a shared per-channel fp32
tensor: lam = 1 / D^2 makes a D-shrunk stream normalise exactly as the original, lam = 1
is the plain RMSNorm. The anneal moves lam = (1 / D^2)^(1 - t) and, with the calibrated
ratio r of each norm, its gain g(t) = g0 * r^-t, from t = 0 (exact) to t = 1 (plain).
Same signatures as the eager `rms_norm` / `rms_norm_mod`."""

import torch
from torch import Tensor

from kohakumoss_kernels.backends.eager._acc import acc, acc_dtype


class WeightedNorms:
    """`rms_norm` / `rms_norm_mod` over a weighted mean square; `lam` is read per call."""

    def __init__(self, dim: int, device):
        self.lam = torch.ones(dim, dtype=torch.float32, device=device)

    def _normed(self, x: Tensor, eps: float) -> Tensor:
        dt = acc_dtype(x)
        xf = x.to(dt)
        ms = (xf.square() * self.lam.to(dt)).mean(-1, keepdim=True)
        return xf * torch.rsqrt(ms + eps)

    def rms_norm(self, x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
        y = self._normed(x, eps)
        if weight is not None:
            y = y * acc(weight, y.dtype)
        return y.to(x.dtype)

    def rms_norm_mod(self, x, weight, shift, scale, seg, eps: float) -> Tensor:
        y = self._normed(x, eps)
        if weight is not None:
            y = y * acc(weight, y.dtype)
        if scale is not None:
            y = y * (1.0 + scale.to(y.dtype).index_select(0, seg))
        if shift is not None:
            y = y + shift.to(y.dtype).index_select(0, seg)
        return y.to(x.dtype)


def anneal_t(step: int, anneal_steps: int) -> float:
    """t of a step: min(1, step / anneal_steps); 1 without an anneal."""
    return 1.0 if anneal_steps <= 0 else min(1.0, step / anneal_steps)


def anneal_lam(d: Tensor, t: float) -> Tensor:
    """(1 / D^2)^(1 - t)."""
    return d.reciprocal().square().pow(1.0 - t)


class GainAnneal:
    """Each norm's stored gain set to g0 * r^-t from its fp32 copy g0 (the gain at t = 0);
    r = 1 for every norm when the run does not anneal the calibrated ratios."""

    def __init__(self):
        self.gains: list[tuple[Tensor, Tensor, float]] = []

    def add(self, stored: Tensor, r: float) -> None:
        self.gains.append((stored, stored.detach().float().clone(), float(r)))

    @torch.no_grad()
    def set(self, t: float) -> None:
        for stored, base, r in self.gains:
            stored.copy_((base * r ** (-t)).to(stored.dtype))

    @torch.no_grad()
    def check(self) -> None:
        """RuntimeError unless t = 1 really divides every gain by its r."""
        self.set(1.0)
        for stored, base, r in self.gains:
            got = float(stored.float().norm() / base.norm())
            if abs(got * r - 1.0) > 0.02:
                raise RuntimeError(
                    f"gain anneal: ratio {got:.4f}, expected {1 / r:.4f}"
                )
        self.set(0.0)
