"""The cached scan and the lazy catalog: warm == cold, rescans touch only what changed,
ids stay stable, a warm open builds no per-row objects, lazy rows equal eager rows,
plans are identical, and the built dataset survives spawn workers."""

import multiprocessing
import os
import pickle
import tracemalloc
from functools import lru_cache

import numpy as np
import pytest

from kohakumoss_data import DataConfig, build_loader
from kohakumoss_data.catalog import table
from kohakumoss_data.catalog.vault import CatalogVault
from kohakumoss_data.format.catalog import Catalog, read_catalog, write_catalog
from kohakumoss_data.importer import import_source
from kohakumoss_data.registry import PLANNERS, SOURCES
from kohakumoss_data.resolution.geometry import Geometry
from kohakumoss_data.scan import manage, paths, walk
from kohakumoss_data.scan.runner import scan_catalog
from kohakumoss_data.sources import base
from kohakumoss_data.types import CatalogRow
from tests._util.negative import must_fail
from tests.data import fixtures

pytestmark = pytest.mark.cpu


def _source(root):
    return SOURCES.get("kohya")(str(root))


def _scan(root, cache, rescan=False):
    path = paths.scan_path(str(cache), {"source": "kohya", "path": str(root)})
    return scan_catalog(_source(root), path, rescan=rescan, workers=1)


def _rows(cat) -> list[CatalogRow]:
    return [cat[i] for i in range(len(cat))]


def _same(a: list, b: list) -> None:
    assert a == b


@pytest.fixture
def probes(monkeypatch):
    """Counts image-header probes (in-process scans)."""
    calls = []
    real = base.probe_image

    def counted(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(base, "probe_image", counted)
    return calls


def test_warm_equals_cold(tmp_path, probes):
    fixtures.make_tree(str(tmp_path / "d"), 3, 6)
    plain = _rows(Catalog(list(_source(tmp_path / "d").scan())))
    first = _scan(tmp_path / "d", tmp_path / "cache")
    warm = _scan(tmp_path / "d", tmp_path / "cache")
    assert _rows(first) == plain and _rows(warm) == plain
    assert warm.counts.probed == 0 and warm.counts.cached == 18
    must_fail(lambda: _same(_rows(warm), plain[::-1]), "catalog in another order")


def _edit_sidecar(path: str) -> None:
    """Same-size content change, mtime moved forward by one second."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    st = os.stat(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.replace("tag_a", "tag_z"))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def _check_edit(root, cache, probes, locator) -> None:
    before = _rows(_scan(root, cache))
    _edit_sidecar(os.path.join(root, locator.replace(".png", ".txt")))
    probes.clear()
    after = _rows(_scan(root, cache))
    assert len(probes) == 1
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(changed) == 1 and after[changed[0]].locator == locator
    assert "tag z" in after[changed[0]].tags or "tag_z" in after[changed[0]].tags


def test_sidecar_edit_rereads_only_its_row(tmp_path, probes, monkeypatch):
    root = tmp_path / "d"
    locators = fixtures.make_tree(str(root), 2, 5)
    _check_edit(root, tmp_path / "cache", probes, locators[3])
    blind = walk.Listing.stat

    def size_only(self, name):
        size, _ = blind(self, name)
        return size, 0

    monkeypatch.setattr(walk.Listing, "stat", size_only)
    root2 = tmp_path / "e"
    locators2 = fixtures.make_tree(str(root2), 2, 5)
    must_fail(
        lambda: _check_edit(root2, tmp_path / "cache2", probes, locators2[3]),
        "a cache that ignores mtime",
    )


def _add_delete(root, cache):
    first = _scan(root, cache)
    ids = {first.locator(i): i for i in range(len(first))}
    os.remove(os.path.join(root, "2_f0", "i001.png"))
    fixtures.write(
        os.path.join(root, "2_f1", "i999.png"), fixtures.png_bytes(20, 20, 7)
    )
    return ids, _scan(root, cache)


def test_added_and_deleted_files_are_reflected(tmp_path, monkeypatch):
    fixtures.make_tree(str(tmp_path / "d"), 2, 4)
    ids, after = _add_delete(str(tmp_path / "d"), tmp_path / "cache")
    gone = ids["2_f0/i001.png"]
    assert not after.live[gone] and after.weight[gone] == 0.0
    assert after.locator(len(after) - 1) == "2_f1/i999.png" and after.live[-1]
    assert int(after.live.sum()) == 8
    monkeypatch.setattr(CatalogVault, "tombstone", lambda self, i: None)
    fixtures.make_tree(str(tmp_path / "e"), 2, 4)

    def kept():
        _, again = _add_delete(str(tmp_path / "e"), tmp_path / "cache2")
        assert int(again.live.sum()) == 8

    must_fail(kept, "a deleted file that stays live")


def test_second_scan_probes_no_header(tmp_path, probes):
    fixtures.make_tree(str(tmp_path / "d"), 2, 6)
    _scan(tmp_path / "d", tmp_path / "cache")
    assert len(probes) == 12
    probes.clear()
    assert _scan(tmp_path / "d", tmp_path / "cache").counts.probes == 0
    assert probes == []

    def forced():
        probes.clear()
        _scan(tmp_path / "d", tmp_path / "cache", rescan=True)
        assert probes == []

    must_fail(forced, "a scan that re-probes everything (DATA_RESCAN=True)")


def test_live_ids_stay_stable_and_compaction_renumbers(tmp_path):
    root = str(tmp_path / "d")
    fixtures.make_tree(root, 2, 4)
    ids, after = _add_delete(root, tmp_path / "cache")
    for i in after.live_ids().tolist():
        loc = after.locator(i)
        if loc in ids:
            assert ids[loc] == i
    path = paths.scan_path(str(tmp_path / "cache"), {"source": "kohya", "path": root})
    del after  # the compaction swaps the file under no open reader
    manage.compact(_source(root), path, workers=1)
    dense = read_catalog(path)
    cold = Catalog(list(_source(root).scan()))
    assert _rows(dense) == _rows(cold) and bool(dense.live.all())

    def stable():
        for i in range(len(dense)):
            if dense.locator(i) in ids:
                assert ids[dense.locator(i)] == i

    must_fail(stable, "ids after compaction (renumbered on purpose)")


def _many(n: int) -> list[CatalogRow]:
    return [
        CatalogRow(
            id=0,
            locator=f"f{i // 1000}/{i:07d}.png",
            w=512 + i % 7,
            h=512,
            group=f"g{i % 13}",
            weight=1.0 + (i % 3),
            tags=["a_tag", "other_tag", f"t{i}"],
            caption_nl="a caption" if i % 2 else None,
            extra={"hash": f"stat:{i}:{i}"},
        )
        for i in range(n)
    ]


def _peak(fn) -> tuple[int, object]:
    tracemalloc.start()
    out = fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak, out


def test_warm_open_builds_no_row_objects(tmp_path, monkeypatch):
    n = 20_000
    path = str(tmp_path / "c.db")
    write_catalog(path, _many(n))
    built = []

    class Counted(CatalogRow):
        def __post_init__(self):
            built.append(1)

    monkeypatch.setattr(table, "CatalogRow", Counted)
    peak, cat = _peak(lambda: read_catalog(path))
    assert len(cat) == n and built == []
    bound = 160 * n
    assert peak < bound, peak

    def eager():
        rows_peak, _ = _peak(lambda: Catalog(list(cat.iter_rows())))
        assert rows_peak < bound, rows_peak

    must_fail(eager, "a catalog of per-row objects")


def test_lazy_row_equals_eager_row(tmp_path, monkeypatch):
    rows = _many(300)
    path = str(tmp_path / "c.db")
    write_catalog(path, rows)
    eager = Catalog(_many(300))
    lazy = read_catalog(path)
    assert _rows(lazy) == _rows(eager)
    real = table.VaultFetch.__call__
    monkeypatch.setattr(
        table.VaultFetch, "__call__", lambda s, i: real(s, (i + 1) % 300)
    )
    shifted = read_catalog(path)
    must_fail(lambda: _same(_rows(shifted), _rows(eager)), "rows fetched off by one")


def _plans(catalog) -> list:
    geometry = Geometry(
        area=256 * 256,
        step=16,
        aspect_range=(0.25, 4.0),
        upscale="never",
        crop="center",
        latent_factor=16,
    )
    planner = PLANNERS.get("bucket")(
        catalog, geometry, seed=3, batching="count", batch_size=4
    )
    return [list(b) for b in planner.epoch(0)]


def test_planner_identical_lazy_and_eager(tmp_path):
    path = str(tmp_path / "c.db")
    write_catalog(path, _many(2000))
    lazy = read_catalog(path)
    assert _plans(lazy) == _plans(Catalog(_many(2000)))
    other = _many(2000)
    for r in other[::7]:
        r.weight = 0.0
    must_fail(lambda: _same(_plans(lazy), _plans(Catalog(other))), "other weights")


def _dataset_cfg(root: str) -> DataConfig:
    fixtures.make_kohya(os.path.join(root, "kohya"))
    fixtures.make_folder(os.path.join(root, "folder"))
    folder = SOURCES.get("folder")(os.path.join(root, "folder"))
    import_source(folder, os.path.join(root, "shard"))
    return DataConfig(
        datasets=[
            {"source": "kohya", "path": os.path.join(root, "kohya")},
            {"source": "folder", "path": os.path.join(root, "folder"), "weight": 0.5},
            {"source": "moss_shard", "path": os.path.join(root, "shard")},
        ],
        cache_dir=os.path.join(root, "cache"),
        latent_cache="off",
        text_cache="off",
        target_area=64 * 64,
        num_workers=2,
    )


def _fetch(ds, ids: list[int]) -> list:
    return [(ds.catalog[i].locator, ds.caption(i, 0)) for i in ids]


def _pickles(obj) -> None:
    try:
        pickle.dumps(obj)
    except Exception as err:
        raise AssertionError(f"not picklable: {err!r}") from err


def test_built_dataset_survives_spawn_workers(tmp_path):
    loader = build_loader(_dataset_cfg(str(tmp_path)))
    ds = loader.dataset
    ids = list(range(len(ds.catalog)))
    want = _fetch(ds, ids)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(2) as pool:
        got = pool.starmap(_fetch, [(ds, ids[:4]), (ds, ids[4:])])
    assert got[0] + got[1] == want
    assert next(iter(loader)).ids.numel() > 0
    _pickles(ds)
    state = dict(ds.catalog.__dict__)
    state["_lru"] = lru_cache(maxsize=8)(ds.catalog.fetch)
    must_fail(lambda: _pickles(state), "a catalog state holding its LRU wrapper")


def test_weights_scale_in_float32():
    part = Catalog(_many(10))
    joined = Catalog.concat([part, part], [1.0, 0.3])
    assert joined.weight.dtype == np.float32
    want = (part.weight * np.float32(0.3)).tolist()
    assert joined.weight[10:].tolist() == want
