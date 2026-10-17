"""The batch layout (SeqInfo) is built in the loader worker when the text lengths are
final, carried with the batch, and used by DeviceBatcher instead of a rebuild."""

import pytest
import torch

from kohakumoss.training import step as step_mod
from kohakumoss.training.step import DeviceBatcher, LayoutTransform, OnlineText
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_data import DataConfig, MossDataset, build_loader
from kohakumoss_data.cache.build import build_text
from kohakumoss_data.cache.random_encoders import RandomTextEncoder
from tests._util.negative import must_fail
from tests.training._train_fakes import host_batch

pytestmark = pytest.mark.cpu
DEV = torch.device("cpu")


def _same_layout(a, b) -> None:
    for name in ("hw", "text_len", "seg_joint", "pos_joint", "img_in_joint"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert a.attn_joint.lengths == b.attn_joint.lengths


def test_transform_attaches_the_layout_of_encoded_text():
    hb = LayoutTransform()(host_batch(0, 0, 16, 32))
    _same_layout(hb.layout, build_seqinfo(hb.hw, hb.text_len))
    pending = host_batch(0, 0, 16, 32)
    pending.text, pending.prompts = None, ["a", "b"]
    assert LayoutTransform()(pending).layout is None


def test_batcher_uses_the_carried_layout(monkeypatch):
    hb = LayoutTransform()(host_batch(0, 0, 16, 32))
    ref = DeviceBatcher(DEV)(host_batch(0, 0, 16, 32)).seq

    def no_rebuild(*args, **kw):
        raise AssertionError("layout rebuilt on the main process")

    monkeypatch.setattr(step_mod, "build_seqinfo", no_rebuild)
    _same_layout(DeviceBatcher(DEV)(hb).seq, ref)
    must_fail(lambda: DeviceBatcher(DEV)(host_batch(0, 0, 16, 32)), "no layout")
    pending = host_batch(0, 0, 16, 32)
    pending.text, pending.prompts = None, ["a", "b"]
    monkeypatch.undo()
    enc = RandomTextEncoder(dim=32)
    seq = DeviceBatcher(DEV, text=OnlineText(enc))(LayoutTransform()(pending)).seq
    assert seq.text_len.tolist() == [t.shape[0] for t in enc(["a", "b"])]


def test_layout_crosses_the_worker_boundary(tmp_path):
    base = dict(
        datasets=[{"source": "synthetic", "count": 8, "seed": 2}],
        resolution_mode="square_crop",
        batching="count",
        batch_size=2,
        target_area=256 * 256,
        latent_cache="off",
        cache_dir=str(tmp_path / "cache"),
        caption_variants=1,
        text_drop=0.0,
        num_workers=1,
    )
    build_text(
        MossDataset(DataConfig(**base, text_cache="offline")), RandomTextEncoder(32)
    )
    loader = build_loader(
        DataConfig(**base, text_cache="offline"), transform=LayoutTransform()
    )
    hb = next(iter(loader))
    assert hb.layout is not None
    _same_layout(hb.layout, build_seqinfo(hb.hw, hb.text_len))
