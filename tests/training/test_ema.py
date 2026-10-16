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


def test_gamma_known_values():
    assert abs(gamma_from_sigma_rel(0.05) - 16.97) < 0.02
    assert abs(gamma_from_sigma_rel(0.10) - 6.94) < 0.02
    for s in (0.02, 0.08, 0.2):
        assert abs(sigma_rel_of(gamma_from_sigma_rel(s)) - s) < 1e-9


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
