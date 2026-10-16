"""Catalog: columnar record table (in memory) and its KohakuVault `catalog.db` form."""

import os
from collections.abc import Iterable
from dataclasses import fields

import kohakuvault as kv
import numpy as np

from kohakumoss_data.types import CatalogRow

COLUMNS = {
    "locator": "str:utf8",
    "w": "i64",
    "h": "i64",
    "has_alpha": "i64",
    "group": "str:utf8",
    "weight": "f64",
    "tags": "msgpack",
    "caption_nl": "msgpack",
    "artist": "msgpack",
    "rating": "msgpack",
    "json": "msgpack",
    "extra": "msgpack",
}


class Catalog:
    """Rows of one or more sources; `source[i]` indexes `sources`, ids are positions."""

    def __init__(self, rows: list[CatalogRow], source: np.ndarray | None = None):
        self.rows = rows
        for i, row in enumerate(rows):
            row.id = i
        self.w = np.array([r.w for r in rows], dtype=np.int64)
        self.h = np.array([r.h for r in rows], dtype=np.int64)
        self.weight = np.array([r.weight for r in rows], dtype=np.float64)
        self.source = np.zeros(len(rows), dtype=np.int64) if source is None else source

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> CatalogRow:
        return self.rows[index]

    @classmethod
    def concat(cls, parts: list["Catalog"], weights: list[float]) -> "Catalog":
        """Join catalogs; part k's row weights are multiplied by `weights[k]`."""
        rows, source = [], []
        for k, (part, scale) in enumerate(zip(parts, weights)):
            for row in part.rows:
                row.weight = row.weight * scale
                rows.append(row)
                source.append(k)
        return cls(rows, np.array(source, dtype=np.int64))


def write_catalog(path: str, rows: Iterable[CatalogRow]) -> int:
    """Write rows to a ColumnVault file at `path` (replacing it); returns the row count."""
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)
    vault = kv.ColumnVault(path)
    cols = {name: vault.ensure(name, dtype) for name, dtype in COLUMNS.items()}
    buffers: dict[str, list] = {name: [] for name in COLUMNS}
    count = 0
    for row in rows:
        for name in COLUMNS:
            value = getattr(row, name)
            if name == "has_alpha":
                value = int(value)
            buffers[name].append(value)
        count += 1
        if count % 4096 == 0:
            _flush(cols, buffers)
    _flush(cols, buffers)
    vault.checkpoint()
    return count


def _flush(cols, buffers) -> None:
    for name, values in buffers.items():
        if values:
            cols[name].extend(values)
            values.clear()


def read_catalog(path: str) -> Catalog:
    """Load a `catalog.db` written by `write_catalog`."""
    vault = kv.ColumnVault(path)
    data = {name: list(vault[name]) for name in COLUMNS}
    names = {f.name for f in fields(CatalogRow)}
    rows = []
    for i in range(len(data["locator"])):
        values = {name: data[name][i] for name in COLUMNS if name in names}
        values["has_alpha"] = bool(values["has_alpha"])
        values["extra"] = values["extra"] or {}
        rows.append(CatalogRow(id=i, **values))
    return Catalog(rows)
