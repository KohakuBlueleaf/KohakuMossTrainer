"""KohakuClip image shards read by key: the index lists every `<key>.<ext>` member with
its byte range and size (the zip64 directory the shard writer stores), keeps a key's
first shard, filters by `keep`, and the store returns each image's exact bytes; a gbc
entry's `images` store is checked."""

import os
import pickle

import numpy as np
import pytest

from kohakumoss_data.images import KcIndex, KcStore, build_index, open_shard, shard_keys
from kohakumoss_data.sources.gbc import GbcSource
from tests._util.negative import must_fail

writer = pytest.importorskip("kohakugbc_data.dataset.images")

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]


def _payload(key: int) -> bytes:
    return bytes((key * 7 + i) % 251 for i in range(50 + key % 37))


def _shard(path, keys, sizes=None) -> None:
    with writer.ImageShardWriter(path, {"source": "test"}) as w:
        for i, key in enumerate(keys):
            wh = sizes[i] if sizes else (10 + key, 20 + key)
            w.add(f"{key}.jpg", _payload(key), wh[0], wh[1], "jpeg")


@pytest.fixture
def shards(tmp_path):
    root = tmp_path / "shards"
    root.mkdir()
    _shard(root / "a.zip", [5, 3, 9])
    _shard(root / "b.zip", [12, 3, 40])
    return str(root)


def _check_store(root, out) -> None:
    store = KcStore(out, root)
    for key in (3, 5, 9, 12, 40):
        assert store.read(key) == _payload(key), key
    index = KcIndex(out)
    row = index.find(12)
    assert index.size_of(row) == (22, 32)
    assert index.shards[int(index.cols["shard"][index.find(3)])] == "a.zip"
    assert index.find(4) < 0


def test_index_and_store(shards, tmp_path):
    out = tmp_path / "index"
    assert build_index(shards, ["a.zip", "b.zip"], out, workers=2) == 5
    _check_store(shards, out)
    keys = np.load(out / "key.npy")
    assert list(keys) == sorted(keys)


def test_index_and_store_negative(shards, tmp_path):
    out = tmp_path / "index"
    build_index(shards, ["a.zip", "b.zip"], out, workers=2)
    off = np.load(out / "off.npy")
    np.save(out / "off.npy", off + 1)
    must_fail(lambda: _check_store(shards, out), "byte offsets off by one")


def test_keep_and_keys(shards, tmp_path):
    out = tmp_path / "kept"
    assert build_index(shards, ["a.zip", "b.zip"], out, keep={3, 40}, workers=1) == 2
    store = KcStore(out, shards)
    assert store.read(40) == _payload(40)
    with pytest.raises(KeyError):
        store.read(5)
    keys, records = shard_keys(open_shard(shards, "b.zip"))
    assert list(keys) == [12, 3, 40] and len(records) == 3


def test_store_pickles(shards, tmp_path):
    out = tmp_path / "index"
    build_index(shards, ["a.zip"], out, workers=1)
    store = pickle.loads(pickle.dumps(KcStore(out, shards)))
    assert store.read(9) == _payload(9)


def test_gbc_images_problems(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    (root / "dataset.json").write_text("{}")
    entry = {"source": "gbc", "path": str(root)}
    assert (
        GbcSource.problems(entry) == [] or "kohakugbc" in GbcSource.problems(entry)[0]
    )
    bad = GbcSource.problems({**entry, "images": {"index": str(tmp_path / "none")}})
    assert bad
    os.makedirs(tmp_path / "idx")
    missing = {"index": str(tmp_path / "idx"), "root": str(tmp_path)}
    assert GbcSource.problems({**entry, "images": missing})
