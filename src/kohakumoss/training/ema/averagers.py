"""Weight EMA as explicit fp32 buffers, updated after each successful optimiser step.

`ConstantEMA` keeps one average per decay; `PowerEMA` keeps EDM2 power-function
averages (beta_t = (1 - 1/t)^(gamma + 1)) per sigma_rel, whose snapshots allow post-hoc
reconstruction (`posthoc.py`).
"""

from functools import partial

import torch
from torch import Tensor, nn


def gamma_from_sigma_rel(sigma_rel: float, iters: int = 60) -> float:
    """Solve sigma_rel^2 = (g+1) / ((g+2)^2 (g+3)) for gamma (EDM2, eq. 3): the largest
    root of g^3 + 7 g^2 + (16 - t) g + 12 - t, t = sigma_rel^-2, by float32 Newton
    from g = sqrt(t), right of the root where the cubic is convex and increasing."""
    t = torch.tensor(sigma_rel, dtype=torch.float32).pow(-2)
    g = t.sqrt()
    for _ in range(iters):
        f = ((g + 7.0) * g + (16.0 - t)) * g + (12.0 - t)
        df = (3.0 * g + 14.0) * g + (16.0 - t)
        g = g - f / df
    return float(g)


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


CHUNK_NUMEL = 64 * 1024 * 1024  # fp32 elements of converted copies alive at once


def _chunks(indices: list[int], sizes: list[int], limit: int) -> list[list[int]]:
    """Consecutive groups of `indices` whose summed size stays within `limit` (a
    tensor larger than `limit` is a group of its own)."""
    out, cur, total = [], [], 0
    for i in indices:
        if cur and total + sizes[i] > limit:
            out.append(cur)
            cur, total = [], 0
        cur.append(i)
        total += sizes[i]
    return out + ([cur] if cur else [])


class _BufferEMA:
    """fp32 average buffers. Parameters already fp32 on the buffer's device are lerped
    in place in one foreach call; the others are converted in chunks of at most
    `chunk_numel` elements, so one chunk of fp32 copies is alive at a time."""

    def __init__(
        self,
        named: list[tuple[str, nn.Parameter]],
        tags,
        device,
        chunk_numel: int = CHUNK_NUMEL,
    ):
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
        ref = next(iter(self.buffers.values()), [])
        same = [
            i
            for i, (p, b) in enumerate(zip(self.params, ref))
            if p.dtype == torch.float32 and p.device == b.device
        ]
        rest = [i for i in range(len(self.params)) if i not in set(same)]
        self.direct = same
        self.converted = _chunks(
            rest, [p.numel() for p in self.params], max(int(chunk_numel), 1)
        )

    def _lerp(self, tag: str, weight: float) -> None:
        bufs = self.buffers[tag]
        if self.direct:
            torch._foreach_lerp_(
                [bufs[i] for i in self.direct],
                [self.params[i].detach() for i in self.direct],
                weight,
            )
        for chunk in self.converted:
            src = [
                self.params[i].detach().to(bufs[i].device, torch.float32) for i in chunk
            ]
            torch._foreach_lerp_([bufs[i] for i in chunk], src, weight)

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
        self,
        named,
        decays: tuple[float, ...],
        warmup: bool = True,
        device=None,
        chunk_numel: int = CHUNK_NUMEL,
    ):
        super().__init__(named, [f"ema{d:g}" for d in decays], device, chunk_numel)
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

    def __init__(
        self,
        named,
        sigma_rels: tuple[float, ...],
        device=None,
        chunk_numel: int = CHUNK_NUMEL,
    ):
        super().__init__(
            named, [f"power{s:g}" for s in sigma_rels], device, chunk_numel
        )
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
