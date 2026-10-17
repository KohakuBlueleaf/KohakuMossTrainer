"""float32 sampling weights: catalog column bytes, legacy read, stable sums, draws."""

import math
import struct

import kohakuvault as kv
import numpy as np
import pytest

from kohakumoss_data.format.catalog import Catalog, read_catalog, write_catalog
from kohakumoss_data.resolution.base import epoch_order, expected_items
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
    raw = list(kv.ColumnVault(path)["weight32"])
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


def test_legacy_weight_column(tmp_path):
    path = str(tmp_path / "old.db")
    write_catalog(path, rows([0.0] * 3))
    vault = kv.ColumnVault(path)
    vault.delete_column("weight32")
    vault.ensure("weight", "f64").extend([2.0, 0.5, 1.25])
    vault.checkpoint()
    assert read_catalog(path).weight.tolist() == [2.0, 0.5, 1.25]


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
