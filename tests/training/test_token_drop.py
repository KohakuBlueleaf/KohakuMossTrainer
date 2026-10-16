"""Random image-token drop: positions are the kept tokens' grid positions, the online
gather keeps exactly those tokens, and a dropped batch trains."""

import pytest
import torch

from kohakumoss.diffusion.objective import MSELoss, TokenNormalizer, no_weighting
from kohakumoss.diffusion.timestep import MossTimestep
from kohakumoss.model import random_backbone
from kohakumoss.training.full import FullTrainable
from kohakumoss.training.optim import build_bundle
from kohakumoss.training.precision import autocast_factory, build_policy
from kohakumoss.training.step import (
    DeviceBatcher,
    OnlineLatents,
    TrainStep,
    gather_kept,
    image_token_count,
)
from kohakumoss.types.seqinfo import build_seqinfo
from tests._util.negative import must_fail
from tests.training._train_fakes import host_batch

pytestmark = pytest.mark.cpu
DEV = torch.device("cpu")
KEEP = [torch.tensor([0, 2, 5]), None]


def _dropped_batch(model):
    hb = host_batch(0, 0, model.cfg.latent_channels, model.cfg.text_dim)
    sizes = (hb.hw[:, 0] * hb.hw[:, 1]).tolist()
    full = hb.latents
    hb.latents = gather_kept(full, hb.hw, KEEP)
    hb.img_keep = KEEP
    return hb, full, sizes


def test_positions_are_original_grid_positions():
    model = random_backbone("tiny", device=DEV, dtype=torch.float32, seed=0)
    hb, _, sizes = _dropped_batch(model)
    batch = DeviceBatcher(DEV)(hb)
    full = build_seqinfo(hb.hw, hb.text_len)
    seq = batch.seq
    assert seq.n_img == 3 + sizes[1] == image_token_count(hb)
    got = seq.pos_joint.index_select(0, seq.img_in_joint)
    full_img = full.pos_joint.index_select(0, full.img_in_joint)
    first = full_img[: sizes[0]].index_select(0, KEEP[0])
    want = torch.cat([first, full_img[sizes[0] :]])
    assert torch.equal(got, want)

    def broken():
        wrong = torch.cat([full_img[:3], full_img[sizes[0] :]])
        assert torch.equal(got, wrong)

    must_fail(broken, "kept tokens given the first grid positions")


def test_online_gather_matches_cached():
    model = random_backbone("tiny", device=DEV, dtype=torch.float32, seed=0)
    hb, full, _ = _dropped_batch(model)
    cached = hb.latents
    hb.latents, hb.pixels = None, [None, None]
    online = OnlineLatents(lambda pixels, hw: full)
    kept, encoded = online(hb, DEV)
    assert torch.equal(kept, cached) and torch.equal(encoded, full)


def test_dropped_batch_trains():
    model = random_backbone("tiny", device=DEV, dtype=torch.float32, seed=0)
    trainable = FullTrainable(model)
    policy = build_policy("fp32", ())
    bundle = build_bundle(
        trainable.param_groups(1e-3, weight_decay=0.0, rules="none"),
        DEV,
        "adamw",
        {"lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8},
        {"mode": "constant", "end": -1},
        20,
        policy,
        1.0,
    )
    step = TrainStep(
        model,
        MSELoss(model.ops.token_mse),
        no_weighting,
        MossTimestep(),
        TokenNormalizer(),
        autocast_factory(policy, DEV),
        DEV,
    )
    hb, _, _ = _dropped_batch(model)
    batch = DeviceBatcher(DEV)(hb)
    denom = step.denominator(batch)
    assert int(denom) == batch.seq.n_img
    losses = []
    for i in range(20):
        bundle.backward(step(batch, denom, i).loss)
        assert bundle.step()
        losses.append(bundle.last_loss)
    assert losses[-1] < losses[0]
