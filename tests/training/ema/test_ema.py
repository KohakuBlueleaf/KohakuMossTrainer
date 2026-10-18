"""EMA recurrences, EDM2 gamma <-> sigma_rel, post-hoc reconstruction."""

import math

import pytest
import torch

from kohakumoss.training.ema import (
    ConstantEMA,
    PowerEMA,
    combine,
    gamma_from_sigma_rel,
    solve_weights,
)
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def sigma_rel_of(gamma: float) -> float:
    return math.sqrt((gamma + 1) / ((gamma + 2) ** 2 * (gamma + 3)))


def _gamma64(sigma_rel: float) -> float:
    """The fp64 companion-matrix reference (test only)."""
    t = sigma_rel**-2
    comp = torch.zeros(3, 3, dtype=torch.float64)
    comp[0] = -torch.tensor([7.0, 16.0 - t, 12.0 - t], dtype=torch.float64)
    comp[1, 0] = comp[2, 1] = 1.0
    eig = torch.linalg.eigvals(comp)
    return max(float(e.real) for e in eig if abs(float(e.imag)) < 1e-9)


def test_gamma_known_values():
    assert abs(gamma_from_sigma_rel(0.05) - 16.97) < 0.02
    assert abs(gamma_from_sigma_rel(0.10) - 6.94) < 0.02
    for s in (0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.25):
        g = gamma_from_sigma_rel(s)
        assert abs(g - _gamma64(s)) / _gamma64(s) < 1e-5, s
        assert abs(sigma_rel_of(g) - s) < 1e-6
    must_fail(
        lambda: _rel_close(gamma_from_sigma_rel(0.05, iters=2), _gamma64(0.05)),
        "Newton not converged",
    )


def _rel_close(a: float, b: float) -> None:
    assert abs(a - b) / abs(b) < 1e-5


def test_constant_ema_recurrence():
    p = torch.nn.Parameter(torch.zeros(3))
    ema = ConstantEMA([("p", p)], (0.9,), warmup=False)
    ref = torch.zeros(3)
    for t in range(10):
        with torch.no_grad():
            p.fill_(float(t))
        ema.update(t)
        ref = 0.9 * ref + 0.1 * float(t)
    assert torch.allclose(ema.averaged("ema0.9")["p"], ref)


def _run_power(gammas_sigma, steps, f):
    p = torch.nn.Parameter(torch.zeros(1))
    ema = PowerEMA([("p", p)], gammas_sigma)
    snaps = {}
    for t in range(steps):
        with torch.no_grad():
            p.fill_(f(t + 1))
        ema.update(t)
        if (t + 1) % 100 == 0:
            snaps[t + 1] = {tag: ema.averaged(tag)["p"].clone() for tag in ema.tags}
    return ema, snaps


def test_power_ema_of_linear_sequence():
    ema, _ = _run_power((0.1,), 4000, lambda t: float(t))
    gamma = ema.gammas[0]
    expect = 4000 * (gamma + 1) / (gamma + 2)
    assert abs(ema.averaged(ema.tags[0])["p"].item() - expect) / expect < 5e-3


def _signal(t: int) -> float:
    return math.sin(t / 300.0) + t / 2000.0


def test_posthoc_reconstructs_unseen_sigma_rel():
    f = _signal
    ema, snaps = _run_power((0.05, 0.10), 2000, f)
    target, _ = _run_power((0.07,), 2000, f)
    entries, states = [], []
    for step, tags in snaps.items():
        for tag, gamma in zip(ema.tags, ema.gammas):
            entries.append((float(step), gamma))
            states.append({"p": tags[tag]})
    w = solve_weights(entries, 2000.0, gamma_from_sigma_rel(0.07))
    got = combine(states, w)["p"].item()
    want = target.averaged(target.tags[0])["p"].item()
    assert abs(got - want) < 2e-3

    def broken():
        naive = states[-1]["p"].item()
        assert abs(naive - want) < 2e-3

    must_fail(broken, "last snapshot used as the target EMA")


def _inner64(ti, gi, tj, gj):
    lo = min(ti, tj)
    return (
        (gi + 1)
        * (gj + 1)
        * lo ** (gi + gj + 1)
        / ((gi + gj + 1) * ti ** (gi + 1) * tj ** (gj + 1))
    )


def _residual64(snaps, x, tt, gt) -> float:
    """|sum x_i p_i - p_t|^2 / |p_t|^2 in fp64 (test only)."""
    x = [float(v) for v in x]
    pt = _inner64(tt, gt, tt, gt)
    quad = sum(
        xi * xj * _inner64(*si, *sj)
        for xi, si in zip(x, snaps)
        for xj, sj in zip(x, snaps)
    )
    lin = sum(xi * _inner64(*si, tt, gt) for xi, si in zip(x, snaps))
    return (quad - 2 * lin + pt) / pt


def _solve64(snaps, tt, gt):
    a = torch.tensor(
        [[_inner64(*si, *sj) for sj in snaps] for si in snaps], dtype=torch.float64
    )
    b = torch.tensor([_inner64(*si, tt, gt) for si in snaps], dtype=torch.float64)
    return torch.linalg.lstsq(a, b.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)


def _inner32(ti, gi, tj, gj):
    """The raw inner-product formula evaluated in fp32 (t^(g+1) overflows)."""
    f = torch.float32
    ti, gi, tj, gj = (torch.tensor(v, dtype=f) for v in (ti, gi, tj, gj))
    lo = torch.minimum(ti, tj)
    num = (gi + 1) * (gj + 1) * lo ** (gi + gj + 1)
    return num / ((gi + gj + 1) * ti ** (gi + 1) * tj ** (gj + 1))


def _naive32(snaps, tt, gt):
    """fp32 solve of the raw Gram system built in fp32."""
    a = torch.stack([torch.stack([_inner32(*si, *sj) for sj in snaps]) for si in snaps])
    b = torch.stack([_inner32(*si, tt, gt) for si in snaps])
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), "fp32 Gram overflow"
    return torch.linalg.lstsq(a, b.unsqueeze(-1), driver="gelsd").solution.squeeze(-1)


@pytest.mark.parametrize("horizon", [2_000, 200_000])
def test_fp32_posthoc_fits_as_well_as_fp64(horizon):
    gammas = [gamma_from_sigma_rel(s) for s in (0.05, 0.10)]
    snaps = [
        (float(t), g)
        for t in range(horizon // 20, horizon + 1, horizon // 20)
        for g in gammas
    ]
    tt, gt = float(horizon), gamma_from_sigma_rel(0.07)
    best = _residual64(snaps, _solve64(snaps, tt, gt), tt, gt)
    got = _residual64(snaps, solve_weights(snaps, tt, gt), tt, gt)
    assert solve_weights(snaps, tt, gt).dtype == torch.float32
    assert got < best + 1e-4, (got, best)

    def naive():
        x = _naive32(snaps, tt, gt)
        assert torch.isfinite(x).all(), "non-finite weights"
        bad = _residual64(snaps, x, tt, gt)
        assert bad < best + 1e-4, (bad, best)

    if horizon == 200_000:
        must_fail(naive, "raw Gram matrix in fp32")
