"""One writer per vault file: per-rank online cache shards (never a shared file), union
reads, merge == union; the catalog lock file and the completion marker (a loser reads
only a complete catalog)."""

import os
import subprocess
import sys
import threading

import pytest
import torch

from kohakumoss_data import CacheWriter, DataConfig, MossDataset
from kohakumoss_data.cache import store as store_mod
from kohakumoss_data.cache.store import (
    TensorStore,
    encode_tensor,
    merge_shards,
    shard_path,
)
from kohakumoss_data.catalog.vault import CatalogVault
from kohakumoss_data.registry import SOURCES
from kohakumoss_data.scan import lock, paths, runner
from kohakumoss_data.scan.runner import scan_catalog
from tests._util.encode_fns import ByteCodes
from tests._util.negative import must_fail
from tests.data import fixtures

pytestmark = pytest.mark.cpu
ENC = ByteCodes(dim=8)


def _online(tmp_path) -> MossDataset:
    cfg = DataConfig(
        datasets=[{"source": "synthetic", "count": 12, "seed": 3}],
        resolution_mode="resize",
        target_area=256 * 256,
        latent_cache="off",
        text_cache="online",
        cache_dir=str(tmp_path / "cache"),
        caption_variants=1,
        text_drop=0.0,
    )
    return MossDataset(cfg)


def _write(ds, rank: int, index: int) -> list[str]:
    """Rank `rank` writes the text encodings of batch `index`; the keys written."""
    writer = CacheWriter(ds, rank=rank)
    hb = ds.materialize(index)
    texts = ENC(hb.prompts)
    writer(hb, None, torch.cat(texts), torch.tensor([t.shape[0] for t in texts]))
    writer.close()
    return list(hb.text_keys)


def _distinct(paths_: list[str]) -> None:
    assert len(set(paths_)) == len(paths_)


def test_two_writers_never_open_the_same_file(tmp_path):
    ds = _online(tmp_path)
    main = ds.texts.store.path
    files = [CacheWriter(ds, rank=r).shard("text") for r in (0, 1)]
    _distinct([main, *files])
    _write(ds, 0, 0)
    _write(ds, 1, 1)
    assert os.path.exists(files[0]) and os.path.exists(files[1])
    assert not os.path.exists(main)
    same = [CacheWriter(ds, rank=0).shard("text") for _ in range(2)]
    must_fail(lambda: _distinct([main, *same]), "two writers on one rank's shard")


def _found(store: TensorStore, keys: list[str]) -> None:
    assert all(k in store and store.get(k) is not None for k in keys)


def test_union_lookup_finds_another_ranks_key(tmp_path, monkeypatch):
    ds = _online(tmp_path)
    keys = _write(ds, 1, 2)
    reader = TensorStore(ds.texts.store.path)
    _found(reader, keys)
    monkeypatch.setattr(store_mod, "shard_paths", lambda path: [])
    must_fail(lambda: _found(TensorStore(ds.texts.store.path), keys), "main file only")


def _view(store: TensorStore, keys: set[str]) -> dict:
    return {k: store.get(k) for k in sorted(keys)}


def _equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    assert all(b[k] is not None and torch.equal(a[k], b[k]) for k in a)


def _shards(main: str) -> set[str]:
    """Ranks 0 and 1 write distinct keys, rank 2 repeats two of rank 0's."""
    keys = set()
    for rank, names in ((0, "abcde"), (1, "f0123"), (2, "ab")):
        writer = TensorStore(main, writes=shard_path(main, rank))
        for name in names:
            key = name * 32
            writer.put(key, torch.full((3 + ord(name) % 4, 8), float(ord(name))))
            keys.add(key)
        writer.flush()
    return keys


def test_merge_equals_union(tmp_path, monkeypatch):
    main = str(tmp_path / "a" / "text.db")
    os.makedirs(os.path.dirname(main))
    keys = _shards(main)
    union = _view(TensorStore(main), keys)
    lengths = TensorStore(main).scan_lengths()
    assert merge_shards(main) == len(keys)
    assert store_mod.shard_paths(main) == []
    _equal(union, _view(TensorStore(main), keys))
    table = TensorStore(main).lengths()
    assert all(table.get(k) == lengths[k] for k in keys)
    other = str(tmp_path / "b" / "text.db")
    os.makedirs(os.path.dirname(other))
    more = _shards(other)
    full = _view(TensorStore(other), more)
    first = store_mod.shard_paths(other)[:1]
    monkeypatch.setattr(store_mod, "shard_paths", lambda path: first)
    merge_shards(other)
    monkeypatch.undo()
    for shard in store_mod.shard_paths(other):
        os.remove(shard)
    merged = _view(TensorStore(other), more)
    must_fail(lambda: _equal(full, merged), "a merge that skips a shard")


def _lengths_known(table, want: dict[str, int]) -> None:
    assert all(table.get(k) == n for k, n in want.items()), want


def test_merge_takes_lengths_from_the_header(tmp_path):
    main = str(tmp_path / "text.db")
    writer = TensorStore(main, writes=shard_path(main, 0))
    writer.put("a" * 32, torch.zeros(5, 4))
    writer.vault["t:" + "b" * 32] = encode_tensor(torch.zeros(7, 4), "bf16")
    writer.flush()
    del writer
    want = {"a" * 32: 5, "b" * 32: 7}
    union = TensorStore(main).lengths()
    must_fail(lambda: _lengths_known(union, want), "an entry without its m: length")
    assert merge_shards(main) == 2
    _lengths_known(TensorStore(main).lengths(), want)


def _kohya(root: str, n: int):
    fixtures.make_tree(root, 2, n // 2)
    return SOURCES.get("kohya")(root)


def test_lock_is_exclusive_and_a_dead_owner_is_stale(tmp_path, monkeypatch):
    path = str(tmp_path / "c.db")
    assert lock.acquire(path)
    assert not lock.acquire(path)
    lock.release(path)
    dead = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    with open(path + lock.LOCK, "w", encoding="utf-8") as fh:
        fh.write(f'{{"pid": {int(dead.stdout)}, "host": "{lock.HOST}"}}')
    assert lock.stale(path) and lock.acquire(path)
    lock.release(path)
    with open(path + lock.LOCK, "w", encoding="utf-8") as fh:
        fh.write(f'{{"pid": {int(dead.stdout)}, "host": "{lock.HOST}"}}')
    monkeypatch.setattr(lock, "pid_alive", lambda pid: True)

    def steal():
        assert lock.acquire(path)

    must_fail(steal, "a stale lock taken for a live one")


def _loser_reads(tmp_path, tag: str) -> int:
    """A writer commits its first chunk and stops; a reader starts; the writer is
    released. Returns the row count the reader saw."""
    root = str(tmp_path / tag)
    source = _kohya(root, 40)
    path = paths.scan_path(
        str(tmp_path / f"cache-{tag}"), {"source": "kohya", "path": root}
    )
    first, release = threading.Event(), threading.Event()
    real = CatalogVault.append

    def held_append(self, rows, items, stamps):
        out = real(self, rows, items, stamps)
        if not first.is_set():
            self.commit(1, "kohya")
            first.set()
            release.wait(30)
        return out

    CatalogVault.append = held_append
    try:
        writer = threading.Thread(
            target=scan_catalog, args=(source, path), kwargs={"workers": 1}
        )
        writer.start()
        first.wait(30)
        seen = {}

        def read():
            seen["n"] = len(scan_catalog(source, path, writer=False, wait=60))

        reader = threading.Thread(target=read)
        reader.start()
        reader.join(1.0)
        release.set()
        writer.join(60)
        reader.join(60)
    finally:
        CatalogVault.append = real
    return seen["n"]


def test_lock_loser_reads_a_complete_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CHUNK", 8)
    assert _loser_reads(tmp_path, "a") == 40
    monkeypatch.setattr(lock, "wait_ready", lambda path, timeout, poll=0.25: {})

    def partial_read():
        assert _loser_reads(tmp_path, "b") == 40

    must_fail(partial_read, "a reader that skips the completion-marker wait")
