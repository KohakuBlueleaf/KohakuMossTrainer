"""MOSS-Image timestep sampler: discrete sigma grid with a truncated logit-normal pmf,
a uniform mix, mass moved to sigma = 1 and a static shift; a staged table switches
parameters by step. See docs/concepts/training.md (Timestep samplers)."""

import bisect
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from kohakumoss.diffusion.timestep.shift import shift_table

TECH_REPORT_STAGE3 = (
    (0, {"mu": 0.0, "s": 1.0, "shift": 3.0, "cutoff": None, "p_full": 0.01}),
    (140_000, {"mu": -1.45, "s": 1.05, "shift": 1.0, "cutoff": None, "p_full": 0.01}),
    (207_000, {"mu": -1.45, "s": 1.05, "shift": 1.0, "cutoff": 0.95, "p_full": 0.0}),
    (274_000, {"mu": -1.09, "s": 0.95, "shift": 1.0, "cutoff": 0.95, "p_full": 0.0}),
)


def _cdf_sf(x: Tensor, mu: float, s: float) -> tuple[Tensor, Tensor]:
    """(P(S <= x), P(S > x)) for S = 1 - sigmoid(Z), x in [0, 1], float32; the
    survival side is its own erfc, not 1 - cdf."""
    x = x.clamp(0.0, 1.0)
    u = (torch.log1p(-x) - torch.log(x) - mu) / (s * math.sqrt(2.0))
    cdf, sf = 0.5 * torch.erfc(u), 0.5 * torch.erfc(-u)
    low, high = x <= 0, x >= 1
    cdf = torch.where(low, 0.0, torch.where(high, 1.0, cdf))
    sf = torch.where(low, 1.0, torch.where(high, 0.0, sf))
    return cdf, sf


def moss_pmf(
    steps: int,
    mu: float,
    s: float,
    cutoff: float | None,
    uniform_mix: float,
    p_full: float,
) -> Tensor:
    """pmf [steps] over sigma_j = j/steps, j = 1..steps (float32, sums to 1). Each
    bin is a difference of cdf values below the median and of survival values above
    it, so no bin is a difference of two numbers near 1."""
    a = 1.0 if cutoff is None else float(cutoff)
    j = torch.arange(steps + 1, dtype=torch.float32)
    cdf, sf = _cdf_sf(a * j / steps, mu, s)
    lower = cdf[1:] <= 0.5
    mass = torch.where(lower, cdf[1:] - cdf[:-1], sf[:-1] - sf[1:])
    mass = mass / cdf[-1].clamp_min(torch.finfo(torch.float32).tiny)
    pmf = (1.0 - uniform_mix) * mass + uniform_mix / steps
    pmf = pmf * (1.0 - p_full)
    pmf[-1] += p_full
    return pmf / pmf.sum()


@dataclass(frozen=True)
class _Stage:
    sigma: Tensor
    cdf: Tensor


def _stage(steps: int, uniform_mix: float, **kw) -> _Stage:
    pmf = moss_pmf(
        steps,
        kw.get("mu", 0.0),
        kw.get("s", 1.0),
        kw.get("cutoff"),
        kw.get("uniform_mix", uniform_mix),
        kw.get("p_full", 0.0),
    )
    grid = torch.arange(1, steps + 1, dtype=torch.float32) / steps
    sigma = shift_table(grid, kw.get("shift", 1.0))
    cdf = pmf.cumsum(0)
    cdf[-1] = 1.0
    return _Stage(sigma, cdf)


class MossTimestep:
    """Sigma sampler following the MOSS-Image recipe; pure function of (generator, step)."""

    def __init__(
        self,
        mu: float = -1.09,
        s: float = 0.95,
        shift: float = 1.0,
        cutoff: float | None = 0.95,
        uniform_mix: float = 0.01,
        p_full: float = 0.0,
        steps: int = 1000,
        schedule: str | list | tuple | None = None,
    ):
        match schedule:
            case None:
                kw = {"mu": mu, "s": s, "shift": shift, "cutoff": cutoff}
                table = ((0, {**kw, "p_full": p_full}),)
            case "tech_report":
                table = TECH_REPORT_STAGE3
            case _:
                table = tuple((int(st), dict(kw)) for st, kw in schedule)
        self.starts = [st for st, _ in table]
        self.stages = [_stage(steps, uniform_mix, **kw) for _, kw in table]
        many = len(self.stages) > 1
        self._sample = MossTimestep._staged if many else MossTimestep._single

    def sample(self, batch: int, generator: torch.Generator, step: int) -> Tensor:
        """sigma [batch] float32 on CPU from the stage active at `step`."""
        return self._sample(self, batch, generator, step)

    def stage_at(self, step: int) -> _Stage:
        return self.stages[bisect.bisect_right(self.starts, step) - 1]

    @staticmethod
    def _draw(stage: _Stage, batch: int, generator: torch.Generator) -> Tensor:
        u = torch.rand(batch, generator=generator)
        idx = torch.searchsorted(stage.cdf, u).clamp_max(stage.cdf.numel() - 1)
        return stage.sigma[idx]

    def _single(self, batch: int, generator: torch.Generator, step: int) -> Tensor:
        """sigma [batch] float32 on CPU (one stage)."""
        return self._draw(self.stages[0], batch, generator)

    def _staged(self, batch: int, generator: torch.Generator, step: int) -> Tensor:
        """sigma [batch] float32 on CPU from the stage active at `step`."""
        return self._draw(self.stage_at(step), batch, generator)
