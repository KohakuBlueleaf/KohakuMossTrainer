"""The sampler equals a restatement of the official MOSSImagePipeline loop: shift-5 grid
snapped to the training grid, terminal sigma 0, timestep 1000 sigma, CFG u + g (v - u)
with the skip on the negative branch only, Euler x + (s_next - s) v."""

import pytest
import torch

from kohakumoss.diffusion.sampling import sample, sample_latents, unpack
from tests._util.negative import must_fail
from tests.model.conftest import oracle_from
from tests.model.test_sampling import _official_sigmas

pytestmark = pytest.mark.cpu
SIZES = [(48, 64), (32, 32)]
SEEDS = (3, 8)
STEPS, CFG, SHIFT = 5, 3.0, 5.0


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _official(oracle, pos, neg, skip):
    """MOSSImageScheduler + __call__ loop per sample (FlowMatchEuler step)."""
    sigmas = torch.cat((_official_sigmas(STEPS, SHIFT), torch.zeros(1)))
    out = []
    for (h, w), seed, p, n in zip(SIZES, SEEDS, pos, neg):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn((1, oracle.cfg.latent_channels, h // 16, w // 16), generator=g)
        for i in range(STEPS):
            t = (sigmas[i] * 1000).reshape(1)
            v = oracle.forward_one(x, t, p[None]).float()
            u = oracle.forward_one(x, t, n[None], skip).float()
            x = x + (sigmas[i + 1] - sigmas[i]) * (u + CFG * (v - u))
        out.append(x)
    return out


def _ours(model, pos, neg, mode="two_pass"):
    res = sample_latents(
        model, pos, neg, SIZES, STEPS, CFG, SHIFT, None, SEEDS, "cpu", mode
    )
    return unpack(res.latents, res.hw)


@pytest.fixture
def setup(tiny):
    g = torch.Generator().manual_seed(1)
    pos = [torch.randn(n, tiny.cfg.text_dim, generator=g) for n in (6, 9)]
    neg = [torch.randn(3, tiny.cfg.text_dim, generator=g)] * 2
    ref = _official(oracle_from(tiny), pos, neg, tuple(tiny.cfg.negative_skip_layers))
    return tiny, pos, neg, ref


def _close(outs, ref):
    return all(_rel(a, b) < 2e-5 for a, b in zip(outs, ref))


def _check(cond):
    assert cond


@pytest.mark.parametrize("mode", ["two_pass", "packed"])
def test_matches_official_loop(setup, mode):
    tiny, pos, neg, ref = setup
    assert _close(_ours(tiny, pos, neg, mode), ref)


def test_shift_ignored_is_caught(setup, monkeypatch):
    tiny, pos, neg, ref = setup
    real = sample.moss_sigmas
    monkeypatch.setattr(sample, "moss_sigmas", lambda steps, shift: real(steps, 1.0))
    must_fail(lambda: _check(_close(_ours(tiny, pos, neg), ref)), "shift ignored")


def _broken_two_pass(swap_base, skip_positive):
    def build(backbone, hw, positive, negative, cfg, skip, device):
        pos_text, pos_len = sample._cat(positive, device)
        neg_text, neg_len = sample._cat(negative, device)
        pos_seq = sample.build_seqinfo(hw, pos_len)
        neg_seq = sample.build_seqinfo(hw, neg_len)
        cond = backbone.bind_forward(skip if skip_positive else ())
        uncond = backbone.bind_forward(() if skip_positive else skip)

        def velocity(x, s):
            v = cond(x, pos_text, s, pos_seq).float()
            u = uncond(x, neg_text, s, neg_seq).float()
            return (v if swap_base else u) + cfg * (v - u)

        return velocity

    return build


@pytest.mark.parametrize("swap_base, skip_positive", [(True, False), (False, True)])
def test_cfg_variants_are_caught(setup, monkeypatch, swap_base, skip_positive):
    tiny, pos, neg, ref = setup
    broken = dict(sample.CFG_MODES)
    broken["two_pass"] = _broken_two_pass(swap_base, skip_positive)
    monkeypatch.setattr(sample, "CFG_MODES", broken)
    must_fail(lambda: _check(_close(_ours(tiny, pos, neg), ref)), "cfg variant")
