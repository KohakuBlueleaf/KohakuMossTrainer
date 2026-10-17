"""Review r1 data findings: plan purity, cached-value rounding, busy vaults, planner
kwargs, worker transform, length index, key coverage, WebDataset members."""

import io
import os
import pickle
import sqlite3
import tarfile
import threading
import time

import kohakuvault as kv
import pytest
import torch

from kohakumoss_data import CacheWriter, DataConfig, Loader, MossDataset
from kohakumoss_data.cache.keys import latent_key
from kohakumoss_data.cache.random_encoders import RandomTextEncoder
from kohakumoss_data.cache.store import (
    STORAGE,
    LengthTable,
    TensorStore,
    decode_tensor,
    encode_tensor,
    stored_value,
)
from kohakumoss_data.registry import SOURCES
from kohakumoss_data.types import CatalogRow, PlanItem
from tests._util.negative import must_fail
from tests.data import fixtures

pytestmark = pytest.mark.cpu

ENC_TXT = RandomTextEncoder(dim=8)


def cfg(tmp_path, **kw) -> DataConfig:
    base = dict(
        datasets=[{"source": "synthetic", "count": 40, "seed": 5}],
        resolution_mode="resize",
        batching="tokens",
        target_area=256 * 256,
        max_tokens=700,
        latent_cache="off",
        text_cache="online",
        cache_dir=str(tmp_path / "cache"),
        caption_variants=1,
        text_drop=0.0,
    )
    base.update(kw)
    return DataConfig(**base)


def plan_of(ds) -> list:
    return [
        [it.row_id for it in ds.planner.plan(i, 0, 1).items]
        for i in range(ds.planner.epoch_length(0))
    ]


def _assert_eq(a, b):
    assert a == b


def test_online_text_plan_is_cache_independent(tmp_path):
    before = plan_of(MossDataset(cfg(tmp_path)))
    ds = MossDataset(cfg(tmp_path))
    for row_id in range(len(ds.catalog)):
        caption = ds.caption(row_id, 0)
        ds.texts.put(ds.texts.key(caption), ENC_TXT([caption])[0])
    _assert_eq(plan_of(MossDataset(cfg(tmp_path))), before)
    live = MossDataset(cfg(tmp_path, text_cache="offline"))
    must_fail(lambda: _assert_eq(plan_of(live), before), "plan from live lengths")


@pytest.mark.parametrize("storage", sorted(STORAGE))
def test_stored_value_matches_store(storage):
    t = torch.randn(37, 24) * 7
    via_store = decode_tensor(encode_tensor(t, storage))
    assert torch.equal(stored_value(t, storage), via_store)
    if torch.cuda.is_available():
        on_gpu = stored_value(t.cuda(), storage).cpu()
        assert torch.equal(on_gpu, via_store)


def test_as_cached_equals_a_hit(tmp_path):
    ds = MossDataset(cfg(tmp_path, latent_cache="online", cache_dtype="fp8"))
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


def _assert_tensor(a, b):
    assert torch.equal(a, b)


def _write_while_locked(tmp_path, busy_timeout: float):
    ds = MossDataset(cfg(tmp_path))
    path = ds.texts.store.path
    TensorStore(path).put("0" * 32, torch.zeros(1, 8))
    writer = CacheWriter(ds, busy_timeout=busy_timeout, store_retries=1)
    held = threading.Event()

    def hold():
        con = sqlite3.connect(path, isolation_level=None)
        con.execute("BEGIN EXCLUSIVE")
        held.set()
        time.sleep(6.5)
        con.execute("COMMIT")
        con.close()

    locker = threading.Thread(target=hold)
    locker.start()
    held.wait()
    hb = ds.materialize(0)
    texts = ENC_TXT(hb.prompts)
    lens = torch.tensor([t.shape[0] for t in texts])
    writer(hb, None, torch.cat(texts), lens)
    try:
        writer.close()
        ok = True
    except RuntimeError:
        ok = False
    locker.join()
    return ok, writer


@pytest.mark.slow
def test_writer_survives_a_locked_vault(tmp_path):
    ok, writer = _write_while_locked(tmp_path, busy_timeout=60.0)
    assert ok and writer.written > 0


class FlakyWriter(CacheWriter):
    """Raises DatabaseBusy on the first `fails` puts, like a vault another rank holds."""

    def __init__(self, *args, fails: int, **kwargs):
        self.fails = fails
        super().__init__(*args, **kwargs)

    def _put(self, store, key, value):
        if self.fails:
            self.fails -= 1
            raise kv.DatabaseBusy("database is locked")
        super()._put(store, key, value)


def _write_flaky(tmp_path, busy_timeout: float):
    ds = MossDataset(cfg(tmp_path))
    writer = FlakyWriter(ds, fails=3, busy_timeout=busy_timeout)
    hb = ds.materialize(0)
    texts = ENC_TXT(hb.prompts)
    writer(hb, None, torch.cat(texts), torch.tensor([t.shape[0] for t in texts]))
    try:
        writer.close()
        ok = True
    except RuntimeError:
        ok = False
    return ok, writer, ds.texts.store.path, list(hb.text_keys)


def test_writer_retries_busy(tmp_path):
    ok, writer, path, keys = _write_flaky(tmp_path / "a", busy_timeout=60.0)
    assert ok and writer.busy_waits == 3 and writer.written == len(keys)
    store = TensorStore(path)
    assert all(k in store for k in keys)
    broken, *_ = _write_flaky(tmp_path / "b", busy_timeout=0.0)
    must_fail(lambda: _assert_eq(broken, True), "no busy retry")


def test_planner_kwargs_override(tmp_path):
    ds = MossDataset(
        cfg(tmp_path, text_cache="off", planner_kwargs={"max_samples": 2, "window": 8})
    )
    sizes = [
        len(ds.planner.plan(i, 0, 1).items) for i in range(ds.planner.epoch_length(0))
    ]
    assert max(sizes) == 2


def tag_pid(batch):
    return batch, os.getpid()


def test_transform_runs_in_worker(tmp_path):
    ds = MossDataset(cfg(tmp_path, text_cache="off"))
    out = []
    for batch, pid in Loader(ds, num_workers=2, pin_memory=False, transform=tag_pid):
        out.append((batch.index, batch.ids.tolist(), pid))
        if len(out) == 4:
            break
    assert [o[0] for o in out] == [0, 1, 2, 3]
    assert all(pid != os.getpid() for *_, pid in out)
    named = Loader(ds, pin_memory=False, transform="tests.data.test_review_r1.tag_pid")
    first, pid = next(iter(named))
    assert first.ids.tolist() == out[0][1] and pid == os.getpid()


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


def test_latent_key_covers_decode_policy(tmp_path):
    row = CatalogRow(id=0, locator="a", w=64, h=64, extra={"hash": "h"})
    item = PlanItem(0, (64, 64), (0, 0, 64, 64))
    base = latent_key(row, item, "vae", "bf16", "white", "lanczos", "mode")
    assert latent_key(row, item, "vae", "bf16", "white", "bicubic", "mode") != base
    assert latent_key(row, item, "vae", "bf16", "white", "lanczos", "sample") != base
    a = MossDataset(cfg(tmp_path, latent_cache="online"))
    b = MossDataset(cfg(tmp_path, latent_cache="online", resample="bicubic"))
    plan = a.planner.plan(0, 0, 1).items[0]
    row0 = a.catalog[plan.row_id]
    assert a.latents.key(row0, plan) != b.latents.key(row0, plan)


def test_webdataset_dotted_members(tmp_path):
    path = tmp_path / "w.tar"
    with tarfile.open(path, "w") as tar:
        for name, data in (
            ("k0.0.jpg", fixtures.png_bytes(16, 16, 1)),
            ("k1.png", fixtures.png_bytes(16, 32, 2)),
            ("k1.seg.png", fixtures.png_bytes(8, 8, 3)),
            ("k2.txt", b"no image"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    src = SOURCES.get("webdataset")(str(path))
    rows = list(src.scan())
    assert [(r.extra["key"], r.w, r.h) for r in rows] == [
        ("k0", 16, 16),
        ("k1", 16, 32),
    ]
    assert src.skipped == ["w.tar:k2"]
