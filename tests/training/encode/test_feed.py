"""Encode feed, model-free: chunk image order and per-item split, a batch finished as a
cache hit carries it, the rank loader's state and generations, the knob checks."""

import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kohakumoss.training.checks.encode import encode_problems
from kohakumoss.training.encode.feed import (
    BATCH,
    RankFeed,
    _Stream,
    finish_batch,
    item_share,
)
from kohakumoss.training.encode.service import image_order, pack, unpack
from kohakumoss.training.encode.streamed import StreamedLoader
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _items():
    return [
        {"pixels": [np.zeros((32, 16, 3), np.uint8)], "hw": [[2, 1]]},
        {"pixels": None, "hw": None},
        {
            "pixels": [
                np.zeros((16, 16, 3), np.uint8),
                np.zeros((32, 16, 3), np.uint8),
            ],
            "hw": [[1, 1], [2, 1]],
        },
    ]


def test_image_order_groups_equal_sizes():
    order = image_order(_items())
    sizes = [tuple(_items()[i]["hw"][j]) for i, j in order]

    def check(sizes):
        assert sizes == sorted(sizes)
        assert len(order) == 3

    check(sizes)
    must_fail(lambda: check([(2, 1), (1, 1), (2, 1)]), "unsorted sizes")


def test_pack_roundtrip_keeps_each_prompt():
    parts = [torch.full((n, 4), float(n)) for n in (3, 1, 2)]
    flat, lengths = pack(parts)
    assert lengths == [3, 1, 2] and flat.shape == (6, 4)
    back = unpack((flat, lengths))
    assert [p[0, 0].item() for p in back] == [3.0, 1.0, 2.0]
    assert unpack(pack([])) == []


def _stream():
    return _Stream("s", SimpleNamespace(dataset=None), [], [], threading.Semaphore(1))


def _hb(limit: list[int], keep=None):
    return SimpleNamespace(
        text=None,
        text_len=None,
        latents=None,
        pixels=[np.zeros((16, 16, 3), np.uint8)],
        prompts=["a", "b"],
        text_limit=limit,
        hw=torch.tensor([[2, 2]]),
        img_keep=keep,
    )


def test_finish_batch_cuts_text_and_gathers_kept_latents():
    hb = _hb([2, 0], [torch.tensor([0, 3])])
    text = pack([torch.ones(5, 4), torch.ones(3, 4)])
    full = torch.arange(4.0).view(4, 1)
    finish_batch(hb, text, full, _stream())

    def check(hb):
        assert hb.text_len.tolist() == [2, 3]
        assert hb.text.shape == (5, 4)
        assert hb.latents.view(-1).tolist() == [0.0, 3.0]
        assert hb.pixels is None

    check(hb)
    raw = _hb([2, 0], [torch.tensor([0, 3])])
    raw.text_len = torch.tensor([5, 3])
    raw.text = torch.ones(8, 4)
    raw.latents = full
    must_fail(lambda: check(raw), "uncut text, full latents")


def test_item_share_splits_chunk_seconds_by_tokens_and_images():
    items = [{"pixels": [0, 0]}, {"pixels": None}]
    payload = {
        "text": [(None, [3]), (None, [1])],
        "te_s": 4.0,
        "vae_s": 2.0,
        "tokens": 4,
        "images": 2,
    }
    a, b = item_share(payload, items, 0), item_share(payload, items, 1)
    assert a["te_s"] == 3.0 and b["te_s"] == 1.0
    assert a["vae_s"] == 2.0 and b["vae_s"] == 0.0
    assert item_share(None, items, 0)["te_s"] == 0.0


def _feed():
    return RankFeed(
        rank=0,
        batches=queue.Queue(),
        validation=None,
        control=queue.Queue(),
        reply=queue.Queue(),
        failed=threading.Event(),
        knobs={},
        start={"seed": 0, "start": 0},
        val_count=0,
    )


def _stats():
    return {"te_s": 1.0, "vae_s": 0.5, "tokens": 8, "images": 1, "passes": 1.0}


def test_streamed_loader_state_and_generation():
    feed = _feed()
    for i in range(3):
        feed.batches.put((BATCH, 0, {"seed": 0, "start": i + 1}, f"g0-{i}", _stats()))
    loader = StreamedLoader(feed, None, pin=False)
    it = iter(loader)
    assert next(it) == "g0-0"

    def check(state):
        assert state == {"seed": 0, "start": 1}

    check(loader.state_dict())
    must_fail(lambda: check({"seed": 0, "start": 0}), "state before the batch")
    assert loader.text.stats["seconds"] == 1.0 and loader.vae.stats["images"] == 1
    feed.reply.put(("gen", 1))
    loader.load_state_dict({"seed": 0, "start": 7})
    assert feed.control.get_nowait() == ("seek", 0, {"seed": 0, "start": 7})
    feed.batches.put((BATCH, 1, {"seed": 0, "start": 8}, "g1-0", _stats()))
    assert next(iter(loader)) == "g1-0"
    assert loader.state_dict() == {"seed": 0, "start": 8}


def _knobs(**kw):
    return {
        "ENCODE_DEVICES": (4, 5),
        "ENCODE_DEPTH": 8,
        "ENCODE_CHUNK": 4,
        "DEVICES": (0, 1),
        "TEXT_DEVICE": "",
        **kw,
    }


def test_encode_problems_catch_overlap_and_bad_values():
    def problems(**kw):
        return encode_problems(_knobs(**kw), arch=None)

    assert encode_problems({"ENCODE_DEVICES": ()}) == []
    overlap = problems(DEVICES=(0, 4))
    assert any("also training" in p for p in overlap)
    assert any("ENCODE_DEPTH" in p for p in problems(ENCODE_DEPTH=0))
    assert any("TEXT_DEVICE" in p for p in problems(TEXT_DEVICE="cuda:4"))
    assert any("device indices" in p for p in problems(ENCODE_DEVICES=("cuda:4",)))
