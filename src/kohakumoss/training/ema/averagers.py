"""Weight EMA as explicit fp32 buffers, updated after each successful optimiser step.

`ConstantEMA` keeps one average per decay; `PowerEMA` keeps EDM2 power-function
averages (beta_t = (1 - 1/t)^(gamma + 1)) per sigma_rel, whose snapshots allow post-hoc
reconstruction (`posthoc.py`).
"""

from functools import partial

import torch
from torch import Tensor, nn


def gamma_from_sigma_rel(sigma_rel: float) -> float:
    """Solve sigma_rel^2 = (g+1) / ((g+2)^2 (g+3)) for gamma (EDM2, eq. 3)."""
    t = sigma_rel**-2
    roots = torch.tensor([1.0, 7.0, 16.0 - t, 12.0 - t], dtype=torch.float64)
    companion = torch.zeros(3, 3, dtype=torch.float64)
    companion[0] = -roots[1:] / roots[0]
    companion[1, 0] = 1.0
    companion[2, 1] = 1.0
    eig = torch.linalg.eigvals(companion)
    real = [float(e.real) for e in eig if abs(float(e.imag)) < 1e-9 and e.real > -1]
    return max(real)


class NoEMA:
    """EMA disabled."""

    tags: tuple[str, ...] = ()

    def update(self, step: int) -> None:
        return None

    def state_dict(self) -> dict[str, Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        return None

    def averaged(self, tag: str) -> dict[str, Tensor]:
        raise KeyError(tag)


class _BufferEMA:
    def __init__(self, named: list[tuple[str, nn.Parameter]], tags, device):
        self.names = [n for n, _ in named]
        self.params = [p for _, p in named]
        self.tags = tuple(tags)
        self.buffers = {
            tag: [
                p.detach().to(device=device or p.device, dtype=torch.float32, copy=True)
                for p in self.params
            ]
            for tag in self.tags
        }

    def _lerp(self, tag: str, weight: float) -> None:
        bufs = self.buffers[tag]
        src = [
            p.detach().to(b.device, torch.float32) for p, b in zip(self.params, bufs)
        ]
        torch._foreach_lerp_(bufs, src, weight)

    def state_dict(self) -> dict[str, Tensor]:
        return {
            f"{tag}/{name}": buf
            for tag in self.tags
            for name, buf in zip(self.names, self.buffers[tag])
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        for tag in self.tags:
            for name, buf in zip(self.names, self.buffers[tag]):
                buf.copy_(state[f"{tag}/{name}"])

    def averaged(self, tag: str) -> dict[str, Tensor]:
        return dict(zip(self.names, self.buffers[tag]))


def _warm_decay(decay: float, t: int) -> float:
    return min(decay, (1 + t) / (10 + t))


def _fixed_decay(decay: float, t: int) -> float:
    return decay


class ConstantEMA(_BufferEMA):
    """ema <- decay * ema + (1 - decay) * p, with optional (1+t)/(10+t) warmup."""

    def __init__(
        self, named, decays: tuple[float, ...], warmup: bool = True, device=None
    ):
        super().__init__(named, [f"ema{d:g}" for d in decays], device)
        self.decays = tuple(decays)
        rule = _warm_decay if warmup else _fixed_decay
        self.rules = tuple(partial(rule, d) for d in self.decays)

    @torch.no_grad()
    def update(self, step: int) -> None:
        t = step + 1
        for tag, rule in zip(self.tags, self.rules):
            self._lerp(tag, 1.0 - rule(t))


class PowerEMA(_BufferEMA):
    """EDM2 power-function EMA per sigma_rel."""

    def __init__(self, named, sigma_rels: tuple[float, ...], device=None):
        super().__init__(named, [f"power{s:g}" for s in sigma_rels], device)
        self.sigma_rels = tuple(sigma_rels)
        self.gammas = tuple(gamma_from_sigma_rel(s) for s in sigma_rels)

    @torch.no_grad()
    def update(self, step: int) -> None:
        t = step + 1
        for tag, gamma in zip(self.tags, self.gammas):
            self._lerp(tag, 1.0 - (1.0 - 1.0 / t) ** (gamma + 1.0))


def build_ema(named, spec, device=None):
    """EMA knob: 0 / None -> off; float or list -> ConstantEMA; dict with kind."""
    match spec:
        case None | 0 | 0.0:
            return NoEMA()
        case float() | int():
            return ConstantEMA(named, (float(spec),), device=device)
        case list() | tuple():
            return ConstantEMA(named, tuple(float(d) for d in spec), device=device)
        case {"kind": "power", **rest}:
            return PowerEMA(
                named, tuple(rest["sigma_rel"]), device=rest.get("device", device)
            )
        case {"kind": "constant", **rest}:
            return ConstantEMA(
                named,
                (
                    tuple(rest["decay"])
                    if isinstance(rest["decay"], (list, tuple))
                    else (rest["decay"],)
                ),
                warmup=rest.get("warmup", True),
                device=rest.get("device", device),
            )
        case _:
            raise ValueError(f"invalid EMA spec {spec!r}")


def ema_is_power(ema) -> bool:
    return isinstance(ema, PowerEMA)
