"""Residual-stream penalties on the joint block outputs, added to the training loss
(STREAM_LOSS; CONCENTRATION_LOSS is its `concentration` term alone):

- `concentration`: the top-k channel share of each block output's energy, mean over the
  blocks;
- `peak`: a soft hinge on each sample's largest stream values, the mean of its top
  `top` fraction of |x| per block in original units (the fp16 fix's stream scale
  undone), `softplus((log m - log cap) / tau)`, mean over blocks, weighted per sample by
  its sigma;
- `rms`: each sample's whole-stream RMS per block, R over all of its tokens and channels
  in original units, pulled to `target` from both sides, `(R / target - 1)^2`, mean over
  blocks, weighted per sample by its sigma.

See docs/concepts/training.md#stream-penalties."""

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kohakumoss.model.compile import compiled_glue

KNOB = "STREAM_LOSS"
LEGACY = "CONCENTRATION_LOSS"
PROBE = "STREAM_LOSS_PROBE"
TERMS = {
    "concentration": {"weight", "k"},
    "peak": {"weight", "cap", "top", "tau", "sigma"},
    "rms": {"weight", "target", "sigma"},
}
SIGMA_WEIGHTS = ("sigma", "one_minus_sigma", "none")
PEAK = {"cap": 32768.0, "top": 1e-4, "tau": 0.25, "sigma": "sigma"}
RMS = {"target": 1.0, "sigma": "none"}
TINY = 1e-30


def channel_energy(h: Tensor) -> Tensor:
    """Each channel's energy of `h` [..., D] summed over every leading position, fp32
    [D]; read in `h`'s own dtype with an fp32 result (no fp32 copy of `h`)."""
    rows = h.reshape(-1, h.shape[-1])
    return torch.linalg.vector_norm(rows, dim=0, dtype=torch.float32).square()


def top_shares(e: Tensor, k: int) -> Tensor:
    """Per row of the channel energies `e` [L, D], the share held by its `k` largest
    channels, fp32 [L]."""
    tiny = torch.finfo(torch.float32).tiny
    return e.topk(k, dim=-1).values.sum(-1) / e.sum(-1).clamp_min(tiny)


def top_share(h: Tensor, k: int) -> Tensor:
    """Share of the energy of `h` [..., D] (summed over every leading position) held by
    its `k` largest channels, in fp32."""
    return top_shares(channel_energy(h)[None], k)[0]


def top_mean_abs(h: Tensor, frac: float) -> Tensor:
    """Mean of the largest `frac` share (at least one) of |h|'s elements, in fp32."""
    flat = h.reshape(-1)
    k = max(1, int(math.ceil(frac * flat.numel())))
    return flat.abs().topk(k).values.float().mean()


def token_rms(h: Tensor, scale: float) -> Tensor:
    """Each row's RMS of `h` [T, D] divided by `scale` (original units), fp32 [T]; the
    norm reads `h` in its own dtype with an fp32 result (no fp32 copy of `h`)."""
    norm = torch.linalg.vector_norm(h, dim=-1, dtype=torch.float32)
    return norm / (math.sqrt(h.shape[-1]) * scale)


def token_log_rms(h: Tensor, scale: float) -> Tensor:
    """log of `token_rms`, fp32 [T]."""
    return token_rms(h, scale).clamp_min(TINY).log()


def sample_rms(r: Tensor, seg: Tensor, samples: int) -> Tensor:
    """Per sample, the RMS over all its tokens and channels from the token RMS `r`
    [..., T] (`seg` [T]: each row's sample): sqrt of the mean of r^2 over the sample's
    rows, fp32 [..., samples]."""
    seg = seg.long()
    shape = (*r.shape[:-1], samples)
    sums = r.new_zeros(shape).index_add(-1, seg, r.square())
    counts = r.new_zeros(samples).index_add(0, seg, torch.ones_like(seg, dtype=r.dtype))
    return (sums / counts.clamp_min(1.0)).clamp_min(TINY).sqrt()


@compiled_glue
def block_stats(
    outs: list[Tensor], scales: tuple[float, ...], energy: bool, rms: bool
) -> tuple[Tensor | None, Tensor | None]:
    """(channel energies [L, D] when `energy`, token RMS in original units [L, T] when
    `rms`) of the block outputs `outs`, one graph for every block under COMPILE."""
    e = torch.stack([channel_energy(h) for h in outs]) if energy else None
    r = torch.stack([token_rms(h, s) for h, s in zip(outs, scales)]) if rms else None
    return e, r


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
    if "rms" in out:
        out["rms"] = {**RMS, **out["rms"]}
    if "concentration" in out:
        out["concentration"] = {"k": 32, **out["concentration"]}
    return out


class StreamPenalty:
    """Forward hooks on the joint blocks keep each block's output while `collecting`
    (checkpoint recomputes in backward are not collected); `take(prepared)` returns
    (the weighted penalty sum, its parts detached, each term unweighted and attached).
    """

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

    def _rms(self, p, term: dict, r: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """(the sigma-weighted mean over samples of the per-block (R / target - 1)^2,
        the metrics: geometric-mean token RMS over every block and its largest
        per-block value, the whole-stream RMS over every block and its largest per-block
        value), from the token RMS `r` [L, T] in original units."""
        target = float(term["target"])
        rd = r.detach()
        geo = rd.clamp_min(TINY).log().mean(-1)
        quad = rd.square().mean(-1).sqrt()
        samples = len(p.seq.attn_joint.lengths)
        per_block = (sample_rms(r, p.seq.seg_joint, samples) / target - 1.0).square()
        per_sample = per_block.mean(0)
        w = sigma_weight(p.sigma, term["sigma"]).to(per_sample.device)
        metrics = {
            "stream_rms": geo.mean().exp(),
            "stream_rms_block_max": geo.max().exp(),
            "stream_rms_q": quad.mean(),
            "stream_rms_q_block_max": quad.max(),
        }
        return (w * per_sample).mean(), metrics

    def take(self, p) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        if not self.outs:
            raise RuntimeError(
                f"{KNOB}: no joint block output was seen (the runner calls block parts "
                "directly)"
            )
        terms, parts = {}, {}
        energy, r = block_stats(
            self.outs, self.scales, "concentration" in self.spec, "rms" in self.spec
        )

        if "concentration" in self.spec:
            k = int(self.spec["concentration"]["k"])
            conc = top_shares(energy, k).mean()
            terms["concentration"] = conc
            parts["concentration"] = conc.detach()
        if "peak" in self.spec:
            hinge, largest = self._peak(p, self.spec["peak"])
            terms["peak"] = hinge
            parts["peak_loss"] = hinge.detach()
            parts["stream_peak"] = largest
        if "rms" in self.spec:
            value, metrics = self._rms(p, self.spec["rms"], r)
            terms["rms"] = value
            parts["rms_loss"] = value.detach()
            parts.update(metrics)
        total = sum(float(self.spec[n]["weight"]) * t for n, t in terms.items())
        self.outs = []
        return total, parts, terms


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
    integer k in [1, dim], cap / tau / target > 0, top in (0, 1], a known sigma
    weighting; not both STREAM_LOSS and CONCENTRATION_LOSS; PARALLEL none / ddp; the
    STREAM_LOSS_PROBE knob."""
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
        match name:
            case "concentration":
                top = term.get("k", 32)
                if isinstance(top, bool) or not isinstance(top, int) or top < 1:
                    p.append(f"{where}['k'] must be an int >= 1, got {top!r}")
                elif dim is not None and top > dim:
                    p.append(f"{where}['k']={top} exceeds the model width {dim}")
            case "peak":
                p += _number(term, "cap", 0.0, where, strict=True)
                p += _number(term, "tau", 0.0, where, strict=True)
                p += _number(term, "top", 0.0, where, strict=True)
                if isinstance(term.get("top"), (int, float)) and term["top"] > 1:
                    p.append(f"{where}['top'] must be <= 1, got {term['top']!r}")
            case "rms":
                p += _number(term, "target", 0.0, where, strict=True)
        if name != "concentration" and term.get("sigma", "sigma") not in SIGMA_WEIGHTS:
            p.append(
                f"{where}['sigma'] must be one of {SIGMA_WEIGHTS}, got {term['sigma']!r}"
            )
    if terms and k.get("PARALLEL", "none") not in ("none", "ddp"):
        p.append(f"{KNOB} runs with PARALLEL 'none' or 'ddp', not {k['PARALLEL']!r}")
    p += probe_problems(k, bool(terms))
    return p


def probe_problems(k: dict, has_terms: bool) -> list[str]:
    """STREAM_LOSS_PROBE: an int >= 0 (0 = off); on, it needs a STREAM_LOSS term and
    PARALLEL 'none'."""
    every = k.get(PROBE, 0)
    if isinstance(every, bool) or not isinstance(every, int) or every < 0:
        return [
            f"{PROBE} must be an int >= 0 (steps between probes, 0 = off), got {every!r}"
        ]
    if every == 0:
        return []
    p = []
    if not has_terms:
        p.append(f"{PROBE} measures the {KNOB} terms: set at least one term")
    if k.get("PARALLEL", "none") != "none":
        p.append(f"{PROBE} runs with PARALLEL 'none', not {k['PARALLEL']!r}")
    return p
