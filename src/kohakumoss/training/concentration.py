"""Concentration penalty: the top-k channel share of residual-stream energy at every joint
block output, averaged over the blocks, added to the step loss with a weight (the
CONCENTRATION_LOSS knob). See docs/concepts/training.md#concentration-penalty."""

from contextlib import contextmanager

import torch
from torch import Tensor, nn

KNOB = "CONCENTRATION_LOSS"


def top_share(h: Tensor, k: int) -> Tensor:
    """Share of the energy of `h` [..., D] (summed over every leading position) held by
    its `k` largest channels, in fp32."""
    e = h.float().reshape(-1, h.shape[-1]).square().sum(0)
    return e.topk(k).values.sum() / e.sum().clamp_min(torch.finfo(torch.float32).tiny)


class ConcentrationPenalty:
    """Forward hooks on the joint blocks keep each block's output while `collecting`
    (checkpoint recomputes in backward are not collected); `take()` returns the mean
    top-k share over the collected blocks."""

    def __init__(self, layers: nn.ModuleList, weight: float, k: int):
        self.weight = float(weight)
        self.k = int(k)
        self.on = False
        self.outs: list[Tensor] = []
        for block in layers:
            block.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        if self.on:
            self.outs.append(output)

    @contextmanager
    def collecting(self):
        self.outs, self.on = [], True
        try:
            yield
        finally:
            self.on = False

    def take(self) -> Tensor:
        if not self.outs:
            raise RuntimeError(
                f"{KNOB}: no joint block output was seen (the runner calls block parts "
                "directly)"
            )
        value = torch.stack([top_share(h, self.k) for h in self.outs]).mean()
        self.outs = []
        return value


def build_penalty(k: dict, backbone) -> ConcentrationPenalty | None:
    """The penalty of CONCENTRATION_LOSS ({} = off) on the backbone's joint blocks."""
    spec = dict(k.get(KNOB) or {})
    if not spec or float(spec.get("weight", 0.0)) == 0.0:
        return None
    return ConcentrationPenalty(backbone.layers, spec["weight"], spec.get("k", 32))


def concentration_problems(k: dict, dim: int | None = None) -> list[str]:
    """CONCENTRATION_LOSS: a dict with `weight` >= 0 and integer `k` >= 1 (at most the
    model width when known); single-process or DDP layouts only."""
    spec = k.get(KNOB) or {}
    if not isinstance(spec, dict):
        return [f"{KNOB} must be a dict like {{'weight': 0.01, 'k': 32}}, got {spec!r}"]
    if not spec:
        return []
    p = []
    unknown = sorted(set(spec) - {"weight", "k"})
    if unknown:
        p.append(f"{KNOB} has unknown fields {unknown}; valid: ['k', 'weight']")
    weight, top = spec.get("weight", 0.0), spec.get("k", 32)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
        p.append(f"{KNOB}['weight'] must be a number >= 0, got {weight!r}")
    if isinstance(top, bool) or not isinstance(top, int) or top < 1:
        p.append(f"{KNOB}['k'] must be an int >= 1, got {top!r}")
    elif dim is not None and top > dim:
        p.append(f"{KNOB}['k']={top} exceeds the model width {dim}")
    if k.get("PARALLEL", "none") not in ("none", "ddp"):
        p.append(f"{KNOB} runs with PARALLEL 'none' or 'ddp', not {k['PARALLEL']!r}")
    return p
