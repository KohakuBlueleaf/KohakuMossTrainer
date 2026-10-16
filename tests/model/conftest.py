"""Fixtures for model tests: tiny random model, the official-forward oracle, batches."""

import pytest
import torch

from kohakumoss.model import random_backbone
from kohakumoss.types.arch import PRESETS
from kohakumoss.types.seqinfo import build_seqinfo
from tests.model._official import OfficialTransformer

HWS = ((3, 5), (4, 4), (2, 6))
TEXT_LENS = (7, 3, 11)


def make_batch(cfg, hws=HWS, text_lens=TEXT_LENS, seed=1, dtype=torch.float32):
    """Packed inputs plus the per-sample [1, C, H, W] / [1, L, D] views of the same data."""
    g = torch.Generator().manual_seed(seed)
    lat = [torch.randn(1, cfg.latent_channels, h, w, generator=g) for h, w in hws]
    txt = [torch.randn(1, n, cfg.text_dim, generator=g) for n in text_lens]
    sigma = torch.rand(len(hws), generator=g)
    img = torch.cat([x.flatten(2).transpose(1, 2)[0] for x in lat]).to(dtype)
    text = torch.cat([t[0] for t in txt]).to(dtype)
    seq = build_seqinfo(torch.tensor(hws), torch.tensor(text_lens))
    return img, text, sigma, seq, lat, txt


def oracle_from(model) -> OfficialTransformer:
    oracle = OfficialTransformer(model.cfg)
    sd = {k: v for k, v in model.state_dict().items() if k != "rope.bands"}
    oracle.load_state_dict(sd, strict=True)
    return oracle.to(next(model.parameters()).dtype)


def oracle_outputs(oracle, lat, txt, sigma, skip=()):
    outs = []
    for i, (x, t) in enumerate(zip(lat, txt)):
        y = oracle.forward_one(x, sigma[i : i + 1] * 1000.0, t, skip)
        outs.append(y[0].flatten(1).transpose(0, 1))
    return torch.cat(outs)


@pytest.fixture
def tiny():
    return random_backbone("tiny", seed=3)


@pytest.fixture
def cfg():
    return PRESETS["tiny"]
