"""TensorStore byte layout, fp8 storage, cached-value rounding, the length index, cache
keys, offline cache build + lookup, online write-back, over-long fits from the cache."""

import pickle
import struct

import numpy as np
import pytest
import torch

from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_data import DataConfig, MossDataset
from kohakumoss_data.cache.build import build_latents, build_text
from kohakumoss_data.cache.keys import latent_key, text_key
from kohakumoss_data.cache.store import (
    STORAGE,
    LengthTable,
    TensorStore,
    decode_tensor,
    encode_tensor,
    stored_value,
)
from kohakumoss_data.types import CatalogRow, PlanItem
from tests._util.encode_fns import ByteCodes, PatchMeans
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def check_layout(data: bytes, t: torch.Tensor) -> None:
    magic, code, ndim, _, scale = struct.unpack_from("<4sBBHf", data, 0)
    shape = struct.unpack_from(f"<{ndim}I", data, 12)
    assert magic == b"KMT1" and code == 1 and scale == 1.0
    assert shape == tuple(t.shape)
    expected = t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
    assert data[12 + 4 * ndim :] == expected


def test_store_layout_bf16():
    t = torch.randn(7, 5, dtype=torch.float32)
    data = encode_tensor(t, "bf16")
    check_layout(data, t)
    must_fail(lambda: check_layout(data, t.flip(0)), "flipped payload")
    assert torch.equal(decode_tensor(data), t.to(torch.bfloat16))


def test_store_fp8_error():
    t = torch.randn(64, 32) * 5
    back = decode_tensor(encode_tensor(t, "fp8"), torch.float32)
    rel = (back - t).abs().max() / t.abs().max()
    assert rel < 0.07
    assert len(encode_tensor(t, "fp8")) < len(encode_tensor(t, "bf16"))


@pytest.mark.parametrize("storage", sorted(STORAGE))
def test_stored_value_matches_store(storage):
    t = torch.randn(37, 24) * 7
    via_store = decode_tensor(encode_tensor(t, storage))
    assert torch.equal(stored_value(t, storage), via_store)
    if torch.cuda.is_available():
        on_gpu = stored_value(t.cuda(), storage).cpu()
        assert torch.equal(on_gpu, via_store)


def _assert_eq(a, b):
    assert a == b


def _assert_tensor(a, b):
    assert torch.equal(a, b)


def test_length_index(tmp_path):
    store = TensorStore(str(tmp_path / "t.db"))
    lengths = {f"{k:032x}": 3 + k % 7 for k in range(500)}
    for key, n in lengths.items():
        store.put(key, torch.zeros(n, 2))
    assert store.write_index() == 500
    table = store.lengths()
    assert all(table.get(k) == n for k, n in lengths.items())
    assert table.get("f" * 32) is None
    assert len(pickle.dumps(table)) < 500 * 32
    store.put("e" * 32, torch.zeros(9, 2))
    stale = store.lengths()
    assert stale.get("e" * 32) is None and store.scan_lengths()["e" * 32] == 9
    rebuilt = LengthTable.from_dict(store.scan_lengths())
    must_fail(lambda: _assert_eq(stale.get("e" * 32), rebuilt.get("e" * 32)), "stale")


def test_key_sensitivity():
    r = CatalogRow(id=0, locator="a", w=100, h=80, extra={"hash": "h1"})
    it = PlanItem(0, (64, 80), (0, 0, 100, 80), False, 0)
    base = latent_key(r, it, "vae", "bf16", "white")
    variants = [
        latent_key(
            CatalogRow(**{**r.__dict__, "extra": {"hash": "h2"}}),
            it,
            "vae",
            "bf16",
            "white",
        ),
        latent_key(r, it, "vae2", "bf16", "white"),
        latent_key(r, it, "vae", "fp8", "white"),
        latent_key(
            r, PlanItem(0, (80, 64), it.crop_ltrb, False, 0), "vae", "bf16", "white"
        ),
        latent_key(
            r,
            PlanItem(0, it.target_hw, (1, 0, 100, 80), False, 0),
            "vae",
            "bf16",
            "white",
        ),
        latent_key(
            r, PlanItem(0, it.target_hw, it.crop_ltrb, True, 0), "vae", "bf16", "white"
        ),
    ]
    assert len(set(variants + [base])) == len(variants) + 1
    assert (
        latent_key(
            r, PlanItem(0, it.target_hw, it.crop_ltrb, False, 3), "vae", "bf16", "white"
        )
        == base
    )
    assert latent_key(r, it, "vae", "bf16", "black") == base
    alpha = CatalogRow(**{**r.__dict__, "has_alpha": True})
    assert latent_key(alpha, it, "vae", "bf16", "white") != latent_key(
        alpha, it, "vae", "bf16", "black"
    )
    assert text_key("a", "te", "bf16") != text_key("b", "te", "bf16")
    assert text_key("a", "te", "bf16") != text_key("a", "te2", "bf16")


def test_latent_key_covers_decode_settings(tmp_path):
    row = CatalogRow(id=0, locator="a", w=64, h=64, extra={"hash": "h"})
    item = PlanItem(0, (64, 64), (0, 0, 64, 64))
    base = latent_key(row, item, "vae", "bf16", "white", "lanczos", "mode")
    assert latent_key(row, item, "vae", "bf16", "white", "bicubic", "mode") != base
    assert latent_key(row, item, "vae", "bf16", "white", "lanczos", "sample") != base
    online = dict(latent_cache="online", text_cache="off")
    a = MossDataset(_cfg(tmp_path, **online))
    b = MossDataset(_cfg(tmp_path, resample="bicubic", **online))
    plan = a.planner.plan(0, 0, 1).items[0]
    row0 = a.catalog[plan.row_id]
    assert a.latents.key(row0, plan) != b.latents.key(row0, plan)


def test_as_cached_equals_a_hit(tmp_path):
    cfg = _cfg(tmp_path, latent_cache="online", text_cache="online", cache_dtype="fp8")
    ds = MossDataset(cfg)
    hw = torch.tensor([[3, 4], [2, 5]])
    full = torch.randn(22, 16) * 3
    rounded = ds.latents.as_cached(full, hw)
    for k, part in enumerate(full.split([12, 10])):
        ds.latents.put(f"{k:032x}", part)
    hits = torch.cat([ds.latents.get(f"{k:032x}") for k in range(2)])
    assert torch.equal(rounded, hits)
    must_fail(lambda: _assert_tensor(full.to(torch.bfloat16), hits), "fresh fp32")
    texts = [torch.randn(5, 8), torch.randn(3, 8)]
    for k, t in enumerate(texts):
        ds.texts.put(f"{k:032x}", t)
    got = [ds.texts.get(f"{k:032x}") for k in range(2)]
    assert all(torch.equal(a, b) for a, b in zip(ds.texts.as_cached(texts), got))


def _cfg(tmp_path, **kw) -> DataConfig:
    base = dict(
        datasets=[{"source": "synthetic", "count": 24, "alpha_every": 5}],
        resolution_mode="bucket",
        batching="count",
        batch_size=3,
        target_area=256 * 256,
        latent_cache="offline",
        text_cache="offline",
        cache_dir=str(tmp_path / "cache"),
        caption_variants=2,
        text_drop=0.3,
        flip=True,
        text_encoder_id=ByteCodes.encoder_id,
    )
    base.update(kw)
    return DataConfig(**base)


def test_offline_cache_roundtrip(tmp_path):
    cfg = _cfg(tmp_path)
    ds = MossDataset(cfg)
    with pytest.raises(KeyError):
        ds.materialize(0)
    enc_img, enc_txt = PatchMeans(channels=16), ByteCodes(dim=32)
    assert build_latents(ds, enc_img, batch_size=4) > 0
    assert build_text(ds, enc_txt, batch_size=8) > 0
    assert build_latents(ds, enc_img) == 0 and build_text(ds, enc_txt) == 0
    ds = MossDataset(cfg)
    for index in range(4):
        b = ds.materialize(index)
        hw = b.hw.long()
        assert b.latents.shape == (int((hw[:, 0] * hw[:, 1]).sum()), 16)
        plan = ds.planner.plan(index, 0, 1)
        direct = enc_img([ds.pixels(it) for it in plan.items], b.hw)
        assert torch.equal(b.latents, direct)
        captions = [
            "" if d else ds.caption(it.row_id, it.caption_variant)
            for it, d in zip(plan.items, b.text_drop.tolist())
        ]
        assert b.text_len.tolist() == [12 + len(c.encode()) for c in captions]
        assert torch.equal(b.text, torch.cat(enc_txt(captions)))
        assert b.text_keys == [ds.texts.key(c) for c in captions]


def test_online_write_back(tmp_path):
    cfg = _cfg(tmp_path, latent_cache="online", text_cache="online")
    ds = MossDataset(cfg)
    enc_img, enc_txt = PatchMeans(channels=16), ByteCodes(dim=32)
    first = ds.materialize(1)
    assert first.latents is None and first.text is None
    ds.write_back(first, enc_img(first.pixels, first.hw), enc_txt(first.prompts))
    again = ds.materialize(1)
    assert torch.equal(again.latents, enc_img(first.pixels, first.hw))
    assert torch.equal(again.text, torch.cat(enc_txt(first.prompts)))
    assert again.pixels is None and again.prompts is None


def _overlong_cfg(tmp_path, overlong, **kw) -> DataConfig:
    return _cfg(
        tmp_path,
        resolution_mode="resize",
        batching="tokens",
        max_tokens=300,
        overlong=overlong,
        text_drop=0.0,
        flip=False,
        token_budget="render",
        **kw,
    )


def _cached(cfg):
    ds = MossDataset(cfg)
    enc_img, enc_txt = PatchMeans(channels=16), ByteCodes(dim=32)
    build_text(ds, enc_txt)
    build_latents(ds, enc_img)
    return MossDataset(cfg), enc_img, enc_txt


def test_overlong_truncate_text_cached(tmp_path):
    ds, _, enc_txt = _cached(_overlong_cfg(tmp_path, "truncate_text"))
    cut = 0
    for index in range(12):
        b = ds.materialize(index)
        plan = ds.planner.plan(index, 0, 1)
        hw = b.hw.long()
        assert int((hw[:, 0] * hw[:, 1]).sum() + b.text_len.sum()) <= 300
        offset = 0
        for it, n in zip(plan.items, b.text_len.tolist()):
            full = enc_txt([ds.caption(it.row_id, it.caption_variant)])[0]
            assert n == (it.text_limit or full.shape[0])
            assert torch.equal(b.text[offset : offset + n], full[:n])
            offset += n
            cut += bool(it.text_limit)
    assert cut > 0


def test_overlong_drop_tokens_cached(tmp_path):
    ds, enc_img, _ = _cached(_overlong_cfg(tmp_path, "drop_tokens"))
    dropped = 0
    for index in range(12):
        b = ds.materialize(index)
        plan = ds.planner.plan(index, 0, 1)
        parts = []
        for it, (h, w), keep in zip(plan.items, b.hw.tolist(), b.img_keep):
            full = enc_img([ds.pixels(it)], torch.tensor([[h, w]]))
            parts.append(full if keep is None else full[keep])
            dropped += keep is not None
            assert (keep is None) == (it.img_keep == 0)
        assert torch.equal(b.latents, torch.cat(parts))
        seq = build_seqinfo(b.hw, b.text_len, img_keep=b.img_keep)
        assert seq.n_img == b.latents.shape[0]
        assert seq.n_img + int(b.text_len.sum()) <= 300
    assert dropped > 0
    again = MossDataset(ds.cfg).materialize(3)
    assert all(
        (a is None and c is None) or torch.equal(a, c)
        for a, c in zip(again.img_keep, ds.materialize(3).img_keep)
    )


def test_overlong_drop_tokens_online(tmp_path):
    cfg = _overlong_cfg(tmp_path, "drop_tokens", latent_cache="online")
    ds = MossDataset(cfg)
    build_text(ds, ByteCodes(dim=32))
    ds = MossDataset(cfg)
    found = False
    for index in range(12):
        b = ds.materialize(index)
        assert b.latents is None and len(b.pixels) == len(b.img_keep)
        for p, (h, w), keep in zip(b.pixels, b.hw.tolist(), b.img_keep):
            assert p.shape[:2] == (h * 16, w * 16)
            found |= keep is not None and keep.numel() < h * w
    assert found


def test_resize_uses_cached_lengths(tmp_path):
    cfg = _cfg(
        tmp_path,
        resolution_mode="resize",
        max_tokens=1024,
        latent_cache="off",
        token_budget="render",
    )
    ds = MossDataset(cfg)
    build_text(ds, ByteCodes(dim=8))
    ds = MossDataset(cfg)
    for row_id in range(5):
        caption = ds.caption(row_id, 1)
        assert ds.text_len(row_id, 1) == 12 + len(caption.encode())


def test_text_drop_rate(tmp_path):
    cfg = _cfg(tmp_path, latent_cache="off", text_cache="off", text_drop=0.3)
    ds = MossDataset(cfg)
    drops, prompts = [], []
    for index in range(60):
        b = ds.materialize(index)
        drops += b.text_drop.tolist()
        prompts += [p for p, d in zip(b.prompts, b.text_drop.tolist()) if d]
    rate = np.mean(drops)
    assert 0.18 < rate < 0.42 and set(prompts) == {""}
