"""Residual-stream penalties on the joint block outputs, added to the training loss
(STREAM_LOSS; CONCENTRATION_LOSS is its `concentration` term alone):

- `concentration`: the top-k channel share of each block output's energy, mean over the
  blocks;
- `peak`: a soft hinge on each sample's largest stream values, the mean of its top
  `top` fraction of |x| per block in original units (the fp16 fix's stream scale
  undone), `softplus((log m - log cap) / tau)`, mean over blocks, weighted per sample by
  its sigma.

See docs/concepts/training.md#stream-penalties."""

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor, nn

KNOB = "STREAM_LOSS"
LEGACY = "CONCENTRATION_LOSS"
TERMS = {
    "concentration": {"weight", "k"},
    "peak": {"weight", "cap", "top", "tau", "sigma"},
}
SIGMA_WEIGHTS = ("sigma", "one_minus_sigma", "none")
PEAK = {"cap": 32768.0, "top": 1e-4, "tau": 0.25, "sigma": "sigma"}


def top_share(h: Tensor, k: int) -> Tensor:
    """Share of the energy of `h` [..., D] (summed over every leading position) held by
    its `k` largest channels, in fp32."""
    e = h.float().reshape(-1, h.shape[-1]).square().sum(0)
    return e.topk(k).values.sum() / e.sum().clamp_min(torch.finfo(torch.float32).tiny)


def top_mean_abs(h: Tensor, frac: float) -> Tensor:
    """Mean of the largest `frac` share (at least one) of |h|'s elements, in fp32."""
    flat = h.reshape(-1)
    k = max(1, int(math.ceil(frac * flat.numel())))
    return flat.abs().topk(k).values.float().mean()


def sigma_weight(sigma: Tensor, mode: str) -> Tensor:
    """Per-sample weight of the peak term: sigma, 1 - sigma or 1."""
    match mode:
        case "sigma":
            return sigma.float()
        case "one_minus_sigma":
            return 1.0 - sigma.float()
    return torch.ones_like(sigma, dtype=torch.float32)


def stream_scales(model) -> tuple[float, ...]:
    """Per joint block, the factor the fp16 fix scales its output stream by (1 without
    a fix): the fold the weights carry, else the online fix's factors."""
    depth = model.cfg.depth
    if getattr(model, "fold_scales", None) is not None:
        cs = tuple(model.fold_scales)
    else:
        cs = tuple(getattr(model, "_online_factors", {}).get("c") or ())
    if not cs:
        return (1.0,) * depth
    return cs * depth if len(cs) == 1 else cs


def spec_of(k: dict) -> dict:
    """The STREAM_LOSS terms of the knobs: STREAM_LOSS, or CONCENTRATION_LOSS as the
    `concentration` term; the peak term's unset fields filled with their defaults."""
    spec = dict(k.get(KNOB) or {})
    legacy = dict(k.get(LEGACY) or {})
    if legacy and not spec:
        spec = {"concentration": legacy}
    out = {
        name: dict(term)
        for name, term in spec.items()
        if float(dict(term).get("weight", 0.0)) > 0
    }
    if "peak" in out:
        out["peak"] = {**PEAK, **out["peak"]}
    if "concentration" in out:
        out["concentration"] = {"k": 32, **out["concentration"]}
    return out


class StreamPenalty:
    """Forward hooks on the joint blocks keep each block's output while `collecting`
    (checkpoint recomputes in backward are not collected); `take(prepared)` returns
    (the weighted penalty sum, its parts detached)."""

    def __init__(self, layers: nn.ModuleList, spec: dict, scales: tuple[float, ...]):
        self.spec = spec
        self.scales = scales
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

    def _peak(self, p, term: dict) -> tuple[Tensor, Tensor]:
        """(the sigma-weighted mean over samples of the per-block hinge mean, the
        largest top-fraction mean in original units)."""
        lengths = list(p.seq.attn_joint.lengths)
        log_cap, tau, top = (
            math.log(float(term["cap"])),
            float(term["tau"]),
            float(term["top"]),
        )
        hinges, largest = [[] for _ in lengths], []
        for h, s in zip(self.outs, self.scales):
            for i, rows in enumerate(h.split(lengths)):
                m = top_mean_abs(rows, top) / s
                hinges[i].append(F.softplus((m.clamp_min(1e-30).log() - log_cap) / tau))
                largest.append(m.detach())
        per_sample = torch.stack([torch.stack(x).mean() for x in hinges])
        w = sigma_weight(p.sigma, term["sigma"]).to(per_sample.device)
        return (w * per_sample).mean(), torch.stack(largest).max()

    def take(self, p) -> tuple[Tensor, dict[str, Tensor]]:
        if not self.outs:
            raise RuntimeError(
                f"{KNOB}: no joint block output was seen (the runner calls block parts "
                "directly)"
            )
        total, parts = None, {}
        if "concentration" in self.spec:
            term = self.spec["concentration"]
            conc = torch.stack([top_share(h, int(term["k"])) for h in self.outs]).mean()
            total = float(term["weight"]) * conc
            parts["concentration"] = conc.detach()
        if "peak" in self.spec:
            term = self.spec["peak"]
            hinge, largest = self._peak(p, term)
            weighted = float(term["weight"]) * hinge
            total = weighted if total is None else total + weighted
            parts["peak_loss"] = hinge.detach()
            parts["stream_peak"] = largest
        self.outs = []
        return total, parts


def build_penalty(k: dict, backbone) -> StreamPenalty | None:
    """The penalty of STREAM_LOSS / CONCENTRATION_LOSS ({} = off) on the backbone's
    joint blocks, its stream scales read from the fp16 fix the backbone carries."""
    spec = spec_of(k)
    if not spec:
        return None
    return StreamPenalty(backbone.layers, spec, stream_scales(backbone))


def _number(
    term: dict, name: str, low: float, name_of: str, strict: bool = False
) -> list[str]:
    value = term.get(name)
    if value is None:
        return []
    bad = isinstance(value, bool) or not isinstance(value, (int, float))
    if bad or (value <= low if strict else value < low):
        op = ">" if strict else ">="
        return [f"{name_of}[{name!r}] must be a number {op} {low}, got {value!r}"]
    return []


def stream_loss_problems(k: dict, dim: int | None = None) -> list[str]:
    """STREAM_LOSS: {term: {fields}} with known terms and fields, weights >= 0, an
    integer k in [1, dim], cap / tau > 0, top in (0, 1], a known sigma weighting; not
    both STREAM_LOSS and CONCENTRATION_LOSS; PARALLEL none / ddp."""
    p = []
    for knob in (KNOB, LEGACY):
        if not isinstance(k.get(knob) or {}, dict):
            p.append(f"{knob} must be a dict, got {k[knob]!r}")
    if p:
        return p
    spec, legacy = dict(k.get(KNOB) or {}), dict(k.get(LEGACY) or {})
    if spec and legacy:
        p.append(
            f"set {KNOB} or {LEGACY} ({LEGACY} is STREAM_LOSS's concentration term), not both"
        )
    terms = spec or ({"concentration": legacy} if legacy else {})
    for name, term in terms.items():
        where = f"{LEGACY}" if not spec else f"{KNOB}[{name!r}]"
        if name not in TERMS:
            p.append(f"{KNOB} has unknown term {name!r}; valid: {sorted(TERMS)}")
            continue
        if not isinstance(term, dict):
            p.append(f"{where} must be a dict, got {term!r}")
            continue
        unknown = sorted(set(term) - TERMS[name])
        if unknown:
            p.append(
                f"{where} has unknown fields {unknown}; valid: {sorted(TERMS[name])}"
            )
        p += _number(term, "weight", 0.0, where)
        if name == "concentration":
            top = term.get("k", 32)
            if isinstance(top, bool) or not isinstance(top, int) or top < 1:
                p.append(f"{where}['k'] must be an int >= 1, got {top!r}")
            elif dim is not None and top > dim:
                p.append(f"{where}['k']={top} exceeds the model width {dim}")
        else:
            p += _number(term, "cap", 0.0, where, strict=True)
            p += _number(term, "tau", 0.0, where, strict=True)
            p += _number(term, "top", 0.0, where, strict=True)
            if isinstance(term.get("top"), (int, float)) and term["top"] > 1:
                p.append(f"{where}['top'] must be <= 1, got {term['top']!r}")
            if term.get("sigma", "sigma") not in SIGMA_WEIGHTS:
                p.append(
                    f"{where}['sigma'] must be one of {SIGMA_WEIGHTS}, got {term['sigma']!r}"
                )
    if terms and k.get("PARALLEL", "none") not in ("none", "ddp"):
        p.append(f"{KNOB} runs with PARALLEL 'none' or 'ddp', not {k['PARALLEL']!r}")
    return p
