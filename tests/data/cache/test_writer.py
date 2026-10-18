"""Online cache write-back (CacheWriter) and the exact token budget for pipelines."""

import threading
import time

import pytest
import torch

from kohakumoss.training.step import DeviceBatcher, OnlineLatents, OnlineText
from kohakumoss_data import CacheWriter, DataConfig, MossDataset, pipeline_problems
from kohakumoss_data.cache.build import build_text
from kohakumoss_data.cache.store import TensorStore
from tests._util.encode_fns import ByteCodes, PatchMeans

pytestmark = pytest.mark.cpu

ENC_IMG = PatchMeans(channels=16)
ENC_TXT = ByteCodes(dim=32)


def cfg(tmp_path, **kw) -> DataConfig:
    base = dict(
        datasets=[{"source": "synthetic", "count": 16, "seed": 2}],
        resolution_mode="square_crop",
        batching="count",
        batch_size=4,
        target_area=256 * 256,
        latent_cache="online",
        text_cache="online",
        cache_dir=str(tmp_path / "cache"),
        caption_variants=1,
        text_drop=0.0,
        text_encoder_id=ByteCodes.encoder_id,
    )
    base.update(kw)
    return DataConfig(**base)


def test_second_epoch_hits(tmp_path):
    ds = MossDataset(cfg(tmp_path))
    writer = CacheWriter(ds)
    batcher = DeviceBatcher(
        torch.device("cpu"),
        text=OnlineText(ENC_TXT),
        latents=OnlineLatents(ENC_IMG),
        on_encoded=writer,
    )
    n = ds.planner.epoch_length(0)
    for index in range(n):
        hb = ds.materialize(index)
        assert hb.pixels is not None and hb.prompts is not None
        batcher(hb)
    writer.close()
    assert writer.written == 2 * len(ds.catalog)
    again = MossDataset(cfg(tmp_path))
    for index in range(n, 2 * n):
        hb = again.materialize(index)
        assert hb.pixels is None and hb.prompts is None
        plan = again.planner.plan(index, 0, 1)
        assert torch.equal(
            hb.latents, ENC_IMG([again.pixels(it) for it in plan.items], hb.hw)
        )
        captions = [again.caption(it.row_id, 0) for it in plan.items]
        assert torch.equal(hb.text, torch.cat(ENC_TXT(captions)))


class SlowWriter(CacheWriter):
    """A CacheWriter whose writes wait for `gate`."""

    def __init__(self, *args, gate: threading.Event, **kwargs):
        self.gate = gate
        super().__init__(*args, **kwargs)

    def _put(self, store, key, value):
        self.gate.wait()
        super()._put(store, key, value)


def test_hook_does_not_wait_for_storage(tmp_path):
    ds = MossDataset(cfg(tmp_path))
    gate = threading.Event()
    writer = SlowWriter(ds, gate=gate)
    hb = ds.materialize(0)
    latents = ENC_IMG(hb.pixels, hb.hw)
    texts = ENC_TXT(hb.prompts)
    lens = torch.tensor([t.shape[0] for t in texts])
    start = time.perf_counter()
    writer(hb, latents, torch.cat(texts), lens)
    assert time.perf_counter() - start < 0.5
    assert writer.pending() > 0 and writer.written == 0
    gate.set()
    writer.close()
    assert writer.written == 2 * len(hb.ids)


def _drop_tokens_dataset(tmp_path, max_tokens: int = 300) -> MossDataset:
    return MossDataset(
        cfg(
            tmp_path,
            resolution_mode="resize",
            batching="tokens",
            max_tokens=max_tokens,
            overlong="drop_tokens",
            datasets=[{"source": "synthetic", "count": 24, "seed": 2}],
        )
    )


def test_token_drop_samples_cached_at_full_grid(tmp_path):
    ds = _drop_tokens_dataset(tmp_path)
    writer = CacheWriter(ds)
    batcher = DeviceBatcher(
        torch.device("cpu"),
        text=OnlineText(ENC_TXT),
        latents=OnlineLatents(ENC_IMG),
        on_encoded=writer,
    )
    seen = []
    for index in range(6):
        hb = ds.materialize(index)
        batcher(hb)
        plan = ds.planner.plan(index, 0, 1)
        for key, keep, item, hw in zip(
            hb.latent_keys, hb.img_keep, plan.items, hb.hw.tolist()
        ):
            seen.append((key, keep, ENC_IMG([ds.pixels(item)], torch.tensor([hw]))))
    writer.close()
    store = TensorStore(ds.latents.store.path)
    assert any(k is not None for _, k, _ in seen)
    for key, _, full in seen:
        assert torch.equal(store.get(key), full)


def test_partial_latents_are_refused(tmp_path):
    ds = MossDataset(cfg(tmp_path))
    writer = CacheWriter(ds)
    hb = ds.materialize(0)
    n0 = int(hb.hw[0, 0] * hb.hw[0, 1])
    hb.img_keep = [torch.arange(0, n0, 3)] + [None] * (len(hb.ids) - 1)
    full = ENC_IMG(hb.pixels, hb.hw)
    sizes = (hb.hw[:, 0] * hb.hw[:, 1]).tolist()
    kept = torch.cat(
        [p if k is None else p[k] for p, k in zip(full.split(sizes), hb.img_keep)]
    )
    writer(hb, kept, None, None)
    writer.close()
    store = TensorStore(ds.latents.store.path)
    for key, keep in zip(hb.latent_keys, hb.img_keep):
        assert (key in store) == (keep is None)


def test_text_cut_is_not_cached(tmp_path):
    ds = MossDataset(cfg(tmp_path))
    writer = CacheWriter(ds)
    hb = ds.materialize(0)
    texts = ENC_TXT(hb.prompts)
    hb.text_limit = [5] + [0] * (len(texts) - 1)
    cut = [texts[0][:5]] + texts[1:]
    lens = torch.tensor([t.shape[0] for t in cut])
    writer(hb, None, torch.cat(cut), lens)
    writer.close()
    store = TensorStore(ds.texts.store.path)
    assert hb.text_keys[0] not in store
    assert all(k in store for k in hb.text_keys[1:])


def test_exact_budget_fits_pipe_shape(tmp_path):
    c = cfg(
        tmp_path,
        resolution_mode="resize",
        batching="tokens",
        max_tokens=700,
        max_samples=2,
        token_budget="exact",
        latent_cache="off",
        text_cache="offline",
        overlong="truncate_text",
        caption_variants=2,
        text_drop=0.2,
    )
    assert pipeline_problems(c) == []
    with pytest.raises(KeyError):
        MossDataset(c).planner.epoch(0)
    build_text(
        MossDataset(cfg(tmp_path, text_cache="offline", caption_variants=2)), ENC_TXT
    )
    ds = MossDataset(c)
    for index in range(2 * ds.planner.epoch_length(0)):
        hb = ds.materialize(index)
        img = int((hb.hw[:, 0] * hb.hw[:, 1]).sum())
        assert img + int(hb.text_len.sum()) <= 700 and len(hb.ids) <= 2
    loose = cfg(tmp_path, batching="count", text_cache="online")
    assert len(pipeline_problems(loose)) == 4
    with pytest.raises(ValueError):
        MossDataset(cfg(tmp_path, token_budget="exact", text_cache="online"))
