"""Random image-token drop: positions are the kept tokens' grid positions and the online
gather keeps exactly those tokens."""

import pytest
import torch

from kohakumoss.training.step import (
    DeviceBatcher,
    OnlineLatents,
    gather_kept,
    image_token_count,
)
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_data.types import HostBatch
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
DEV = torch.device("cpu")
KEEP = [torch.tensor([0, 2, 5]), None]
LATENT_CHANNELS, TEXT_DIM = 16, 32


def _host_batch(hw=((2, 3), (3, 2)), text=(5, 4)) -> HostBatch:
    """A packed two-sample HostBatch of seeded random latents and text features."""
    g = torch.Generator().manual_seed(0)
    hw_t = torch.tensor(hw, dtype=torch.int32)
    n_img = int((hw_t[:, 0] * hw_t[:, 1]).sum())
    return HostBatch(
        ids=torch.arange(len(hw)),
        hw=hw_t,
        text_len=torch.tensor(text, dtype=torch.int32),
        text_drop=torch.zeros(len(hw), dtype=torch.bool),
        weight=torch.ones(len(hw)),
        latents=torch.randn(n_img, LATENT_CHANNELS, generator=g),
        text=torch.randn(int(sum(text)), TEXT_DIM, generator=g),
    )


def _dropped_batch():
    hb = _host_batch()
    sizes = (hb.hw[:, 0] * hb.hw[:, 1]).tolist()
    full = hb.latents
    hb.latents = gather_kept(full, hb.hw, KEEP)
    hb.img_keep = KEEP
    return hb, full, sizes


def test_positions_are_original_grid_positions():
    hb, _, sizes = _dropped_batch()
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
    hb, full, _ = _dropped_batch()
    cached = hb.latents
    hb.latents, hb.pixels = None, [None, None]
    online = OnlineLatents(lambda pixels, hw: full)
    kept, encoded = online(hb, DEV)
    assert torch.equal(kept, cached) and torch.equal(encoded, full)
