"""CONCENTRATION_LOSS: the top-k share, the hook collection window and the knob checks."""

import pytest
import torch
from torch import nn

from kohakumoss.training.concentration import (
    ConcentrationPenalty,
    concentration_problems,
    top_share,
)
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]


def _share_check(fn) -> None:
    """Equal channels hold k/d of the energy; one channel 100x the rest holds almost all;
    tokens with a large first token do not move a channel share."""
    flat = torch.ones(16, 64)
    assert fn(flat, 8).item() == pytest.approx(8 / 64)
    spike = torch.ones(16, 64)
    spike[:, 3] = 100.0
    assert fn(spike, 1).item() == pytest.approx(10000 / (10000 + 63), rel=1e-6)
    tokens = torch.ones(16, 64)
    tokens[0] *= 30.0
    assert fn(tokens, 8).item() == pytest.approx(8 / 64)


def test_top_share():
    _share_check(top_share)


def test_top_share_negative():
    def over_tokens(h, k):
        e = h.float().reshape(-1, h.shape[-1]).square().sum(1)
        return e.topk(k).values.sum() / e.sum()

    must_fail(lambda: _share_check(over_tokens), "share over tokens, not channels")


def test_penalty_collects_only_inside_the_window():
    layers = nn.ModuleList(nn.Identity() for _ in range(3))
    pen = ConcentrationPenalty(layers, 0.01, 2)
    x = torch.randn(5, 16)
    for layer in layers:
        layer(x)
    assert pen.outs == []
    with pen.collecting():
        for layer in layers:
            layer(x)
    for layer in layers:
        layer(x)
    assert len(pen.outs) == 3
    assert pen.take().item() == pytest.approx(top_share(x, 2).item())
    with pytest.raises(RuntimeError):
        pen.take()


def test_knob_problems():
    ok = {"CONCENTRATION_LOSS": {"weight": 0.01, "k": 32}, "PARALLEL": "ddp"}
    assert concentration_problems(ok, dim=3072) == []
    assert concentration_problems({"CONCENTRATION_LOSS": {}}) == []
    bad = [
        {"CONCENTRATION_LOSS": {"weight": -1.0}},
        {"CONCENTRATION_LOSS": {"weight": 0.01, "k": 0}},
        {"CONCENTRATION_LOSS": {"weight": 0.01, "topk": 8}},
        {"CONCENTRATION_LOSS": {"weight": 0.01}, "PARALLEL": "pipeline"},
        {"CONCENTRATION_LOSS": 0.01},
    ]
    for k in bad:
        assert concentration_problems(k), k
    assert concentration_problems({"CONCENTRATION_LOSS": {"k": 4096}}, dim=3072)
