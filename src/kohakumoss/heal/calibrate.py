"""Massive-channel calibration on heal records: per-channel stream energy at every reader
norm, the shrink scales D per tau, each norm's per-token divisor ratio, and what the
training-free shrink does to the stream and the velocity. One report (JSON) holds it
all; the heal reads D and the ratios from it."""

from collections import defaultdict

import numpy as np
import torch

from kohakumoss.heal.channels import (
    peak_ratio,
    ratio_summary,
    scaled,
    shrink_scales,
    shrink_table,
    top_channels,
)
from kohakumoss.heal.losses import rel_sq
from kohakumoss.heal.sites import NormTaps


def _forward(model, x, autocast):
    with torch.no_grad(), autocast():
        return model(*x["x"]).float()


def channel_mean_square(model, taps: NormTaps, xs, autocast) -> dict[str, np.ndarray]:
    """Per site: per-channel mean square of the stream over every token of `xs`."""
    sq, n = defaultdict(lambda: 0.0), defaultdict(int)

    def fn(site, x):
        sq[site] = sq[site] + x.float().square().sum(0).cpu().numpy()
        n[site] += x.shape[0]

    taps.fn = fn
    for x in xs:
        _forward(model, x, autocast)
    taps.fn = None
    return {k: sq[k] / n[k] for k in sq}


def divisor_ratios(
    model, taps: NormTaps, xs, ds: dict[float, torch.Tensor], autocast
) -> dict:
    """Per tau and site: the per-token ratio rms(x) / rms(D x), image and text tokens
    apart (the image-token rows of the joint stream from the batch's SeqInfo)."""
    found = {tau: defaultdict(lambda: {"img": [], "txt": []}) for tau in ds}
    dev = {tau: d.to(xs[0]["target"].device) for tau, d in ds.items()}
    state = {"seq": None}

    def fn(site, x):
        xf = x.float()
        base = xf.square().mean(-1)
        img = torch.zeros(xf.shape[0], dtype=torch.bool, device=xf.device)
        match site[0]:
            case "j":
                img[state["seq"].img_in_joint] = True
            case "h":
                img[:] = True
        for tau, d in dev.items():
            r = (base / (xf * d).square().mean(-1)).sqrt()
            found[tau][site]["img"].append(r[img].cpu())
            found[tau][site]["txt"].append(r[~img].cpu())

    taps.fn = fn
    for x in xs:
        state["seq"] = x["x"][3]
        _forward(model, x, autocast)
    taps.fn = None
    out = {}
    for tau, per in found.items():
        out[tau] = {}
        for site, groups in per.items():
            cat = {g: torch.cat(v) if v else torch.empty(0) for g, v in groups.items()}
            out[tau][site] = ratio_summary(cat["img"], cat["txt"])
    return out


def site_stats(
    model, taps: NormTaps, xs, autocast
) -> tuple[dict[str, list[float]], list[torch.Tensor]]:
    """(per site: median over samples of [rms, max |x|, top-8 share], the velocities)."""
    stats = defaultdict(list)

    def fn(site, x):
        xf = x.float()
        e = xf.square().sum(0)
        stats[site].append(
            [
                float(xf.square().mean().sqrt()),
                float(xf.abs().max()),
                float(e.topk(8).values.sum() / e.sum()),
            ]
        )

    taps.fn = fn
    vs = [_forward(model, x, autocast) for x in xs]
    taps.fn = None
    return {k: np.median(np.array(v), axis=0).tolist() for k, v in stats.items()}, vs


@torch.no_grad()
def apply_live(model, table: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply a parameter-name -> row-multiplier table in place; returns the originals."""
    params = dict(model.named_parameters())
    saved = {}
    for name, m in table.items():
        p = params[name]
        saved[name] = p.detach().clone()
        p.copy_(scaled(p, m))
    return saved


@torch.no_grad()
def restore_live(model, saved: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    for name, t in saved.items():
        params[name].copy_(t)


def calibrate(model, taps: NormTaps, calib, held, taus, autocast) -> dict:
    """The calibration report: `massive` (top 32 channels by M_c), `original` (per-site
    stats on `held`), and per tau: `shrunk` [[channel, D_c]], `ratios` (per site rbar and
    quantiles, on `calib`), the shrunk model's per-site stats and per-sample velocity
    error against the original on `held`."""
    sites = taps.sites
    text_blocks, depth = len(model.text_embedder.blocks), model.cfg.depth
    peak = peak_ratio(channel_mean_square(model, taps, calib, autocast))
    ds = {tau: shrink_scales(peak, tau) for tau in taus}
    ratios = divisor_ratios(model, taps, calib, ds, autocast)
    st0, v0 = site_stats(model, taps, held, autocast)
    report = {"massive": top_channels(peak), "original": {"sites": st0}, "taus": {}}
    for tau in taus:
        d = ds[tau]
        rbar = {s: ratios[tau][s]["rbar"] for s in sites}
        saved = apply_live(model, shrink_table(text_blocks, depth, sites, d, rbar))
        st1, v1 = site_stats(model, taps, held, autocast)
        restore_live(model, saved)
        report["taus"][str(float(tau))] = {
            "shrunk": [
                [int(c), round(float(d[c]), 6)]
                for c in (d < 1).nonzero().flatten().tolist()
            ],
            "ratios": ratios[tau],
            "sites": st1,
            "v_rel_err": [float(rel_sq(b, a).sqrt()) for a, b in zip(v0, v1)],
            "sigma": [x["sigma"] for x in held],
        }
    return report


def report_scales(
    report: dict, tau: float, dim: int
) -> tuple[torch.Tensor, dict[str, float]]:
    """(D [dim], per-site rbar) of `tau` in a calibration report."""
    rec = report["taus"][str(float(tau))]
    d = torch.ones(dim)
    for c, v in rec["shrunk"]:
        d[c] = v
    return d, {s: v["rbar"] for s, v in rec["ratios"].items()}
