"""float32 sampling weights: catalog column bytes, stable sums, draws; a catalog file
of another layout is refused naming the import command, and the scan cache rebuilds
one by a rescan."""

import math
import struct

import kohakuvault as kv
import numpy as np
import pytest
from PIL import Image

from kohakumoss_data.catalog.vault import is_catalog_vault
from kohakumoss_data.format.catalog import Catalog, read_catalog, write_catalog
from kohakumoss_data.resolution.base import epoch_order, expected_items
from kohakumoss_data.scan.runner import scan_catalog
from kohakumoss_data.sources.folder import FolderSource
from kohakumoss_data.types import CatalogRow
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

WEIGHTS = [1.0, 0.1, 3.5, 16777215.0, 0.0, 2.75]


def rows(weights) -> list[CatalogRow]:
    return [
        CatalogRow(id=0, locator=str(i), w=8, h=8, weight=w)
        for i, w in enumerate(weights)
    ]


def check_column(path: str, weights) -> None:
    """Witness the `weight32` column against independently packed float32 bytes."""
    vault = kv.ColumnVault(path, read_only=True)
    raw = list(vault["weight32"])
    vault.close()
    assert raw == [struct.pack("<f", w) for w in weights]


def test_catalog_weights_float32(tmp_path):
    path = str(tmp_path / "c.db")
    write_catalog(path, rows(WEIGHTS))
    check_column(path, WEIGHTS)
    must_fail(lambda: check_column(path, WEIGHTS[::-1]), "reversed weights")
    cat = read_catalog(path)
    assert cat.weight.dtype == np.float32
    assert cat.weight.tolist() == np.array(WEIGHTS, dtype=np.float32).tolist()
    assert Catalog(rows(WEIGHTS)).weight.dtype == np.float32


LEGACY = {
    "locator": "str:utf8",
    "w": "i64",
    "h": "i64",
    "has_alpha": "i64",
    "group": "str:utf8",
    "tags": "msgpack",
    "caption_nl": "msgpack",
    "artist": "msgpack",
    "rating": "msgpack",
    "json": "msgpack",
    "extra": "msgpack",
}


def legacy_catalog(path: str, weight_col: tuple[str, str], weights: list) -> None:
    """A catalog.db of the old layout: one ColumnVault column per field, the weights
    in `weight_col` (name, dtype)."""
    vault = kv.ColumnVault(path)
    n = len(weights)
    values = {
        "locator": [str(i) for i in range(n)],
        "w": [8] * n,
        "h": [8] * n,
        "has_alpha": [0] * n,
        "group": ["g"] * n,
        "extra": [{}] * n,
    }
    for name, dtype in LEGACY.items():
        vault.ensure(name, dtype).extend(values.get(name, [None] * n))
    vault.ensure(*weight_col).extend(weights)
    vault.checkpoint()


def _refused(path: str) -> None:
    try:
        read_catalog(path)
    except ValueError as err:
        assert "scripts/data/catalog_build.py (import_to" in str(err), err
        return
    raise AssertionError(f"{path} was read")


def test_another_layout_is_refused_naming_the_import(tmp_path):
    path = str(tmp_path / "old.db")
    legacy_catalog(path, ("weight", "f64"), [2.0, 0.5, 1.25])
    _refused(path)
    assert not is_catalog_vault(path)
    current = str(tmp_path / "c.db")
    write_catalog(current, rows(WEIGHTS))
    must_fail(lambda: _refused(current), "the current layout")
    with pytest.raises(FileNotFoundError):
        read_catalog(str(tmp_path / "absent.db"))


def test_the_scan_cache_rebuilds_another_layout(tmp_path, caplog):
    folder = tmp_path / "images"
    folder.mkdir()
    Image.new("RGB", (16, 16)).save(folder / "a.png")
    path = str(tmp_path / "cache" / "catalog.db")
    (tmp_path / "cache").mkdir()
    legacy_catalog(path, ("weight", "f64"), [2.0, 0.5])
    with caplog.at_level("WARNING", logger="kohakumoss_data"):
        catalog = scan_catalog(FolderSource(str(folder)), path)
    assert int(catalog.live.sum()) == 1 and catalog[0].locator == "a.png"
    assert "has another layout" in caplog.text


def naive_items(weights: np.ndarray) -> float:
    """Sequential float32 accumulation (the unstable form)."""
    return float(np.cumsum(weights.astype(np.float32), dtype=np.float32)[-1])


def check_sum(fn, weights: np.ndarray, rel: float = 1e-6) -> None:
    exact = math.fsum(weights.astype(np.float32).tolist())
    assert abs(fn(weights) - exact) <= rel * exact, (fn(weights), exact)


@pytest.mark.slow
def test_expected_items_stable_for_large_catalogs():
    n = 10_000_000
    frac = np.full(n, 0.1, dtype=np.float32)
    check_sum(expected_items, frac)
    must_fail(lambda: check_sum(naive_items, frac), "sequential float32 sum")
    mixed = np.random.default_rng(0).random(n, dtype=np.float32) * 5
    check_sum(expected_items, mixed)


def test_epoch_counts_float32():
    big = np.array([4194303.0, 3.0, 0.0], dtype=np.float32)
    order = epoch_order(big, 0, 0)
    assert np.bincount(order, minlength=3).tolist() == [4194303, 3, 0]
    half = np.full(200_000, 0.5, dtype=np.float32)
    n = len(epoch_order(half, 1, 0))
    assert abs(n - 100_000) < 5 * math.sqrt(200_000 * 0.25)
    rare = np.full(1_000_000, 1e-3, dtype=np.float32)
    k = len(epoch_order(rare, 2, 0))
    assert abs(k - 1000) < 5 * math.sqrt(1000)
