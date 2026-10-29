"""STREAM_LOSS: the top-k share, the soft top-k peak hinge (sigma weighting, stream-scale
undo, per-sample split), the hook collection window, the spec and the knob checks."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kohakumoss.training.stream_loss import (
    StreamPenalty,
    sigma_weight,
    spec_of,
    stream_loss_problems,
    stream_scales,
    top_mean_abs,
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


def test_top_mean_abs():
    h = torch.zeros(10, 100)
    h[0, :3] = torch.tensor([-50.0, 40.0, 30.0])
    assert top_mean_abs(h, 2e-3).item() == pytest.approx(45.0)
    assert top_mean_abs(h, 1e-9).item() == pytest.approx(50.0)


def _prepared(lengths, sigma):
    attn = SimpleNamespace(lengths=tuple(lengths))
    return SimpleNamespace(
        seq=SimpleNamespace(attn_joint=attn), sigma=torch.tensor(sigma)
    )


def _peak_value(scales, sigma_mode="sigma"):
    """Two samples (4 and 6 joint rows), two blocks; sample 1's peak 4x the cap."""
    layers = nn.ModuleList(nn.Identity() for _ in range(2))
    spec = {
        "peak": {
            "weight": 1.0,
            "cap": 100.0,
            "top": 0.05,
            "tau": 0.25,
            "sigma": sigma_mode,
        }
    }
    pen = StreamPenalty(layers, spec, scales)
    h = torch.ones(10, 8) * scales[0]
    h[5, 0] = 400.0 * scales[0]
    with pen.collecting():
        for layer in layers:
            layer(h)
    total, parts = pen.take(_prepared([4, 6], [0.2, 0.8]))
    return total.item(), parts


def test_peak_hinge():
    total, parts = _peak_value((1.0, 1.0))
    small = math.log1p(math.exp((math.log(1.0) - math.log(100.0)) / 0.25))
    large = math.log1p(
        math.exp((math.log(400.0 * 1 / 3 + 2 / 3) - math.log(100.0)) / 0.25)
    )
    want = (0.2 * small + 0.8 * large) / 2
    assert total == pytest.approx(want, rel=1e-4)
    assert parts["stream_peak"].item() == pytest.approx(400.0 / 3 + 2 / 3, rel=1e-4)
    folded, _ = _peak_value((1 / 16, 1 / 16))
    assert folded == pytest.approx(total, rel=1e-4)


def test_peak_hinge_negative():
    total, _ = _peak_value((1.0, 1.0))
    must_fail(
        lambda: _assert_close(_peak_value((1.0, 1.0), "one_minus_sigma")[0], total),
        "sigma weighting ignored",
    )


def _assert_close(a, b):
    assert a == pytest.approx(b, rel=1e-4)


def test_sigma_weight():
    s = torch.tensor([0.25, 0.75])
    assert sigma_weight(s, "sigma").tolist() == [0.25, 0.75]
    assert sigma_weight(s, "one_minus_sigma").tolist() == [0.75, 0.25]
    assert sigma_weight(s, "none").tolist() == [1.0, 1.0]


def test_penalty_collects_only_inside_the_window():
    layers = nn.ModuleList(nn.Identity() for _ in range(3))
    pen = StreamPenalty(layers, {"concentration": {"weight": 0.5, "k": 2}}, (1.0,) * 3)
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
    total, parts = pen.take(_prepared([5], [0.5]))
    assert parts["concentration"].item() == pytest.approx(top_share(x, 2).item())
    assert total.item() == pytest.approx(0.5 * top_share(x, 2).item())
    with pytest.raises(RuntimeError):
        pen.take(_prepared([5], [0.5]))


def test_stream_scales():
    m = SimpleNamespace(cfg=SimpleNamespace(depth=3), fold_scales=(0.0625,) * 3)
    assert stream_scales(m) == (0.0625,) * 3
    m = SimpleNamespace(
        cfg=SimpleNamespace(depth=3),
        fold_scales=None,
        _online_factors={"c": (0.5, 0.25, 0.125)},
    )
    assert stream_scales(m) == (0.5, 0.25, 0.125)
    m = SimpleNamespace(cfg=SimpleNamespace(depth=2), fold_scales=None)
    assert stream_scales(m) == (1.0, 1.0)


def test_spec():
    assert spec_of({}) == {}
    assert spec_of({"CONCENTRATION_LOSS": {"weight": 0.01}}) == {
        "concentration": {"k": 32, "weight": 0.01}
    }
    got = spec_of(
        {
            "STREAM_LOSS": {
                "peak": {"weight": 0.02, "cap": 1000.0},
                "concentration": {"weight": 0.0},
            }
        }
    )
    assert (
        set(got) == {"peak"}
        and got["peak"]["cap"] == 1000.0
        and got["peak"]["top"] == 1e-4
    )


def test_knob_problems():
    ok = {
        "STREAM_LOSS": {
            "concentration": {"weight": 0.01, "k": 32},
            "peak": {"weight": 0.02, "top": 1e-4},
        },
        "PARALLEL": "ddp",
    }
    assert stream_loss_problems(ok, dim=3072) == []
    assert stream_loss_problems({"CONCENTRATION_LOSS": {"weight": 0.01, "k": 32}}) == []
    assert stream_loss_problems({"STREAM_LOSS": {}}) == []
    bad = [
        {"STREAM_LOSS": {"concentration": {"weight": -1.0}}},
        {"STREAM_LOSS": {"concentration": {"weight": 0.01, "k": 0}}},
        {"STREAM_LOSS": {"scale": {"weight": 0.01}}},
        {"STREAM_LOSS": {"peak": {"weight": 0.01, "cap": 0}}},
        {"STREAM_LOSS": {"peak": {"weight": 0.01, "top": 2.0}}},
        {"STREAM_LOSS": {"peak": {"weight": 0.01, "sigma": "sqrt"}}},
        {"STREAM_LOSS": {"peak": {"weight": 0.01, "topk": 8}}},
        {"STREAM_LOSS": {"peak": {"weight": 0.01}}, "PARALLEL": "pipeline"},
        {
            "STREAM_LOSS": {"peak": {"weight": 0.01}},
            "CONCENTRATION_LOSS": {"weight": 0.01},
        },
        {"STREAM_LOSS": 0.01},
        {"CONCENTRATION_LOSS": {"weight": 0.01, "topk": 8}},
    ]
    for k in bad:
        assert stream_loss_problems(k), k
    assert stream_loss_problems(
        {"CONCENTRATION_LOSS": {"weight": 0.1, "k": 4096}}, dim=3072
    )
