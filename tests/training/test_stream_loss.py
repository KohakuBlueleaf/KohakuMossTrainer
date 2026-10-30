"""STREAM_LOSS: the top-k share, the soft top-k peak hinge and the whole-stream RMS pull
to the target (sigma weighting, stream-scale undo, per-sample split; every row moved by
the same fraction, zero at the target, growing with the distance on both sides), the hook
collection window, the spec and the knob checks."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from kohakumoss.training.stream_loss import (
    StreamPenalty,
    sigma_weight,
    spec_of,
    stream_loss_problems,
    stream_scales,
    token_log_rms,
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
    seg = torch.cat(
        [torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lengths)]
    )
    return SimpleNamespace(
        seq=SimpleNamespace(attn_joint=attn, seg_joint=seg), sigma=torch.tensor(sigma)
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
    total, parts, _ = pen.take(_prepared([4, 6], [0.2, 0.8]))
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
    total, parts, terms = pen.take(_prepared([5], [0.5]))
    assert set(terms) == {"concentration"} and terms[
        "concentration"
    ].item() == pytest.approx(top_share(x, 2).item())
    assert parts["concentration"].item() == pytest.approx(top_share(x, 2).item())
    assert total.item() == pytest.approx(0.5 * top_share(x, 2).item())
    with pytest.raises(RuntimeError):
        pen.take(_prepared([5], [0.5]))


def test_token_log_rms():
    h = torch.zeros(3, 4)
    h[0] = 2.0
    h[1] = torch.tensor([4.0, 0.0, 0.0, 0.0])
    h[2] = 1e-40
    got = token_log_rms(h, 1.0).tolist()
    assert got[0] == pytest.approx(math.log(2.0))
    assert got[1] == pytest.approx(math.log(2.0))
    assert got[2] == pytest.approx(math.log(1e-30))
    half = token_log_rms(h.half() * 0.0625, 0.0625)
    assert half.dtype == torch.float32
    assert half[:2].tolist() == pytest.approx([math.log(2.0)] * 2, rel=1e-3)


def _rms_value(scales, sigma_mode="none", target=1.0):
    """Two samples (2 and 3 rows), two blocks: sample 0's rows at RMS 4 and 1 in block 0
    and RMS 1 in block 1; sample 1 at RMS 8 in both blocks (original units)."""
    layers = nn.ModuleList(nn.Identity() for _ in range(2))
    spec = {"rms": {"weight": 1.0, "target": target, "sigma": sigma_mode}}
    pen = StreamPenalty(layers, spec, scales)
    b0 = torch.tensor([4.0, 1.0, 8.0, 8.0, 8.0]).unsqueeze(1).expand(5, 6).clone()
    b1 = torch.tensor([1.0, 1.0, 8.0, 8.0, 8.0]).unsqueeze(1).expand(5, 6).clone()
    with pen.collecting():
        layers[0](b0 * scales[0])
        layers[1](b1 * scales[1])
    total, parts, terms = pen.take(_prepared([2, 3], [0.25, 0.75]))
    return total.item(), parts, terms


def _rms_want(w0: float, w1: float, target: float = 1.0) -> float:
    """Sample 0's whole-stream RMS is sqrt((16 + 1) / 2) in block 0 and 1 in block 1;
    sample 1's is 8 in both."""
    s0 = ((math.sqrt(8.5) / target - 1) ** 2 + (1 / target - 1) ** 2) / 2
    s1 = (8 / target - 1) ** 2
    return (w0 * s0 + w1 * s1) / 2


def test_rms_term():
    total, parts, terms = _rms_value((1.0, 1.0))
    assert total == pytest.approx(_rms_want(1.0, 1.0), rel=1e-5)
    assert terms["rms"].item() == pytest.approx(total, rel=1e-6)
    assert parts["rms_loss"].item() == pytest.approx(total, rel=1e-6)
    log_means = [
        (math.log(4.0) + 0 + 3 * math.log(8.0)) / 5,
        (0 + 0 + 3 * math.log(8.0)) / 5,
    ]
    assert parts["stream_rms"].item() == pytest.approx(
        math.exp(sum(log_means) / 2), rel=1e-5
    )
    assert parts["stream_rms_block_max"].item() == pytest.approx(
        math.exp(log_means[0]), rel=1e-5
    )
    quad = [math.sqrt((16 + 1 + 3 * 64) / 5), math.sqrt((1 + 1 + 3 * 64) / 5)]
    assert parts["stream_rms_q"].item() == pytest.approx(sum(quad) / 2, rel=1e-5)
    assert parts["stream_rms_q_block_max"].item() == pytest.approx(quad[0], rel=1e-5)
    folded, _, _ = _rms_value((0.0625, 0.0625))
    assert folded == pytest.approx(total, rel=1e-5)
    weighted, _, _ = _rms_value((1.0, 1.0), "sigma")
    assert weighted == pytest.approx(_rms_want(0.25, 0.75), rel=1e-5)
    shifted, _, _ = _rms_value((1.0, 1.0), target=8.0)
    assert shifted == pytest.approx(_rms_want(1.0, 1.0, 8.0), rel=1e-5)


def test_rms_term_negative():
    def token_value():
        """The mean over each sample's tokens of (r / target - 1)^2: every token pulled
        to the target on its own."""
        rows = [[[4.0, 1.0], [8.0] * 3], [[1.0, 1.0], [8.0] * 3]]
        per_sample = [
            sum(sum((r - 1) ** 2 for r in rows[b][i]) / len(rows[b][i]) for b in (0, 1))
            / 2
            for i in (0, 1)
        ]
        return sum(per_sample) / 2

    must_fail(
        lambda: _assert_close(token_value(), _rms_want(1.0, 1.0)),
        "every token pulled to the target",
    )


def _module_loss(h: Tensor) -> Tensor:
    """The rms term (target 1) of one sample whose block output is `h`."""
    layers = nn.ModuleList([nn.Identity()])
    spec = {"rms": {"weight": 1.0, "target": 1.0, "sigma": "none"}}
    pen = StreamPenalty(layers, spec, (1.0,))
    with pen.collecting():
        layers[0](h)
    return pen.take(_prepared([h.shape[0]], [0.5]))[2]["rms"]


def _pull(loss_of, scale: float) -> tuple[Tensor, Tensor]:
    """A row at RMS 1 and a row at RMS 7 (whole-stream RMS 5), scaled by `scale`:
    (rows, gradient of `loss_of`)."""
    h = torch.tensor([[1.0, -1.0, 1.0, 1.0], [7.0, 7.0, -7.0, 7.0]]) * scale
    h.requires_grad_(True)
    (g,) = torch.autograd.grad(loss_of(h), h)
    return h.detach(), g


def _pulls_to_target(loss_of) -> None:
    """Above the target every row is pulled in by the same fraction of itself, below it
    pushed out; zero gradient at the target; the farther the whole-stream RMS from the
    target (either side), the larger the gradient."""
    for scale, sign in ((1.0, 1.0), (0.02, -1.0)):
        h, g = _pull(loss_of, scale)
        rel = [float(torch.dot(r, x) / r.square().sum()) for r, x in zip(h, g)]
        assert sign * rel[0] > 0 and rel[1] == pytest.approx(rel[0], rel=1e-4)
    _, g = _pull(loss_of, 0.2)
    assert float(g.norm()) < 1e-5
    above = [float(_pull(loss_of, s)[1].norm()) for s in (0.4, 1.0, 4.0, 40.0)]
    below = [float(_pull(loss_of, s)[1].norm()) for s in (0.15, 0.1, 0.05, 0.01)]
    assert above == sorted(above) and below == sorted(below)


def test_rms_pulls_the_whole_stream_to_target():
    _pulls_to_target(_module_loss)


def test_rms_pulls_the_whole_stream_to_target_negative():
    def log_square(h):
        return h.square().mean().sqrt().log().square()

    must_fail(lambda: _pulls_to_target(log_square), "(log R)^2: pull fades far above")


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
    assert spec_of({"STREAM_LOSS": {"rms": {"weight": 1e-3}}}) == {
        "rms": {"weight": 1e-3, "target": 1.0, "sigma": "none"}
    }


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
    rms = {"STREAM_LOSS": {"rms": {"weight": 1e-3, "target": 1.0, "sigma": "sigma"}}}
    assert stream_loss_problems(rms) == []
    assert stream_loss_problems({**rms, "STREAM_LOSS_PROBE": 5}) == []
    assert stream_loss_problems({"STREAM_LOSS_PROBE": 0}) == []
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
        {"STREAM_LOSS": {"rms": {"weight": 1e-3, "target": 0.0}}},
        {"STREAM_LOSS": {"rms": {"weight": 1e-3, "sigma": "sqrt"}}},
        {"STREAM_LOSS": {"rms": {"weight": 1e-3, "cap": 1.0}}},
        {"STREAM_LOSS_PROBE": 5},
        {"STREAM_LOSS_PROBE": -1},
        {"STREAM_LOSS_PROBE": True},
        {
            "STREAM_LOSS": {"rms": {"weight": 1e-3}},
            "STREAM_LOSS_PROBE": 5,
            "PARALLEL": "ddp",
        },
    ]
    for k in bad:
        assert stream_loss_problems(k), k
    assert stream_loss_problems(
        {"CONCENTRATION_LOSS": {"weight": 0.1, "k": 4096}}, dim=3072
    )
