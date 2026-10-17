"""Catalog: columnar record table (in memory) and its KohakuVault `catalog.db` form.

Row weights are float32: column `weight32`, 4 little-endian bytes per row.
"""

import os
import struct
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
    "weight32": "bytes:4",
    "tags": "msgpack",
    "caption_nl": "msgpack",
    "artist": "msgpack",
    "rating": "msgpack",
    "json": "msgpack",
    "extra": "msgpack",
}
WEIGHT = struct.Struct("<f")


def _encode(name: str, row: CatalogRow):
    match name:
        case "has_alpha":
            return int(row.has_alpha)
        case "weight32":
            return WEIGHT.pack(row.weight)
        case _:
            return getattr(row, name)


def _weights(vault: kv.ColumnVault) -> list[float]:
    """Row weights; a catalog written without `weight32` holds them in `weight`."""
    names = {column[0] for column in vault.list_columns()}
    if "weight32" in names:
        raw = b"".join(vault["weight32"])
        return np.frombuffer(raw, dtype="<f4").tolist()
    return [float(w) for w in vault["weight"]]


class Catalog:
    """Rows of one or more sources; `source[i]` indexes `sources`, ids are positions.
    `weight` is float32 [N]."""

    def __init__(self, rows: list[CatalogRow], source: np.ndarray | None = None):
        self.rows = rows
        for i, row in enumerate(rows):
            row.id = i
        self.w = np.array([r.w for r in rows], dtype=np.int64)
        self.h = np.array([r.h for r in rows], dtype=np.int64)
        self.weight = np.array([r.weight for r in rows], dtype=np.float32)
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
            buffers[name].append(_encode(name, row))
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
    data = {name: list(vault[name]) for name in COLUMNS if name != "weight32"}
    data["weight"] = _weights(vault)
    names = {f.name for f in fields(CatalogRow)}
    rows = []
    for i in range(len(data["locator"])):
        values = {name: data[name][i] for name in data if name in names}
        values["has_alpha"] = bool(values["has_alpha"])
        values["extra"] = values["extra"] or {}
        rows.append(CatalogRow(id=i, **values))
    return Catalog(rows)
