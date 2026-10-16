"""Backbone vs the official forward: per-sample oracle, packing, skip layers, fp16 fold."""

import pytest
import torch

from kohakumoss.model.numerics import fold, unfold
from kohakumoss.types.seqinfo import build_seqinfo
from tests._util.negative import must_fail
from tests.model.conftest import (
    HWS,
    TEXT_LENS,
    make_batch,
    oracle_from,
    oracle_outputs,
)

pytestmark = pytest.mark.cpu


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _assert_close(out, ref, tol=2e-5):
    assert torch.isfinite(out).all()
    assert ref.abs().max() > 0
    assert _rel(out, ref) < tol, _rel(out, ref)


def test_matches_official(tiny):
    img, text, sigma, seq, lat, txt = make_batch(tiny.cfg)
    out = tiny(img, text, sigma, seq)
    ref = oracle_outputs(oracle_from(tiny), lat, txt, sigma)
    assert out.shape == (seq.n_img, tiny.cfg.latent_channels)
    _assert_close(out, ref)


def test_text_length_drives_image_positions(tiny):
    img, text, sigma, _, lat, txt = make_batch(tiny.cfg)
    ref = oracle_outputs(oracle_from(tiny), lat, txt, sigma)
    wrong = build_seqinfo(torch.tensor(HWS), torch.tensor(TEXT_LENS))
    wrong.pos_joint[: HWS[0][0] * HWS[0][1], 1] = 0.0
    must_fail(lambda: _assert_close(tiny(img, text, sigma, wrong), ref), "f-axis")


def test_packed_equals_per_sample(tiny):
    img, text, sigma, seq, _, _ = make_batch(tiny.cfg)
    packed = tiny(img, text, sigma, seq)
    parts, i0, t0 = [], 0, 0
    for i, ((h, w), n) in enumerate(zip(HWS, TEXT_LENS)):
        one = build_seqinfo(torch.tensor([[h, w]]), torch.tensor([n]))
        parts.append(
            tiny(img[i0 : i0 + h * w], text[t0 : t0 + n], sigma[i : i + 1], one)
        )
        i0, t0 = i0 + h * w, t0 + n
    _assert_close(packed, torch.cat(parts), tol=1e-6)


def test_skip_layers_match_official(tiny):
    img, text, sigma, seq, lat, txt = make_batch(tiny.cfg)
    skip = frozenset(tiny.cfg.negative_skip_layers)
    out = tiny.bind_forward(skip)(img, text, sigma, seq)
    ref = oracle_outputs(oracle_from(tiny), lat, txt, sigma, tuple(skip))
    _assert_close(out, ref)
    must_fail(lambda: _assert_close(tiny(img, text, sigma, seq), ref), "skip ignored")


@pytest.mark.parametrize(
    "scales, via",
    [(2.0**-6, "gate"), (2.0**-6, "proj"), ([2.0**-k for k in (2, 5, 3, 7)], "proj")],
)
def test_fold_is_exact(tiny, scales, via):
    img, text, sigma, seq, _, _ = make_batch(tiny.cfg)
    tiny.double()
    img, text = img.double(), text.double()
    ref = tiny(img, text, sigma, seq)
    before = {k: v.clone() for k, v in tiny.state_dict().items()}
    fold(tiny, scales, via)
    _assert_close(tiny(img, text, sigma, seq), ref, tol=1e-12)
    unfold(tiny)
    for k, v in tiny.state_dict().items():
        assert torch.equal(v, before[k]), k
    _assert_close(tiny(img, text, sigma, seq), ref, tol=1e-12)


def test_skip_keeps_fold_rescale(tiny):
    img, text, sigma, seq, _, _ = make_batch(tiny.cfg)
    tiny.double()
    img, text = img.double(), text.double()
    ref = tiny.bind_forward({1})(img, text, sigma, seq)
    fold(tiny, [2.0**-k for k in (2, 6, 3, 5)], "proj")
    _assert_close(tiny.bind_forward({1})(img, text, sigma, seq), ref, tol=1e-12)
    plan = tiny._plan(frozenset({1}))
    tiny._plan = lambda skip: tuple(
        s for s in plan if not hasattr(s, "args") or s.args[:1] != (2.0**-4,)
    )
    must_fail(
        lambda: _assert_close(
            tiny.bind_forward({1})(img, text, sigma, seq), ref, 1e-12
        ),
        "rescale dropped",
    )


def test_sub_blocks_compose_to_block(tiny):
    img, text, sigma, seq, _, _ = make_batch(tiny.cfg)
    captured = {}

    class Capture:
        def run(self, i, block, x, ctx):
            captured.setdefault(i, (x, ctx))
            return block(x, ctx)

    tiny.set_runner(Capture())
    tiny(img, text, sigma, seq)
    x, ctx = captured[1]
    attn, ffn = tiny.layers[1].sub_blocks()
    assert torch.equal(ffn(attn(x, ctx), ctx), tiny.layers[1](x, ctx))
    assert not torch.equal(attn(x, ctx), tiny.layers[1](x, ctx))


def test_fold_needs_eps_rescale(tiny):
    img, text, sigma, seq, _, _ = make_batch(tiny.cfg)
    tiny.double()
    img, text = img.double(), text.double()
    ref = tiny(img, text, sigma, seq)
    fold(tiny, 2.0**-12, "gate")
    for name, eps in tiny._fold_eps.items():
        tiny.get_submodule(name).eps = eps
    must_fail(lambda: _assert_close(tiny(img, text, sigma, seq), ref, 1e-8), "eps")
