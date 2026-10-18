"""CatalogVault: one catalog file (a ColumnVault and a KVault on one path), single
writer, many readers. Layout: docs/concepts/data.md#the-catalog-file."""

import os
import struct

import kohakuvault as kv
import numpy as np

from kohakumoss_data.catalog.text import CHAR_FIELDS, text_chars
from kohakumoss_data.types import CatalogRow

SCHEMA = 2
STAMPS = (
    "st_size",
    "st_mtime",
    "s1_size",
    "s1_mtime",
    "s2_size",
    "s2_mtime",
    "s3_size",
    "s3_mtime",
)
PLANNING = ("w", "h", "has_alpha", "live", "group_id", *CHAR_FIELDS)
FIXED = (*PLANNING, *STAMPS)
VAR_FIELDS = ("locator", "tags", "caption_nl", "artist", "rating", "json", "extra")
INFO = b"meta:info"
GROUPS = b"meta:groups"
NO_STAMP = (-1,) * len(STAMPS)
ROW = struct.Struct(">q")
WEIGHT = struct.Struct("<f")
PACK = kv.DataPacker("msgpack")


def row_key(i: int) -> bytes:
    """The KVault key of row `i`'s variable fields."""
    return b"r" + ROW.pack(i)


def var_fields(row: CatalogRow) -> dict:
    """The variable (non-column) fields of `row` by name."""
    return {name: getattr(row, name) for name in VAR_FIELDS}


def _raw(col, n: int) -> bytes:
    """The first `n` elements of a fixed-width column as raw bytes (one range read)."""
    if n == 0:
        return b""
    return bytes(
        col._inner.read_range(col._col_id, 0, n, col._elem_size, col._chunk_bytes)
    )


def is_catalog_vault(path: str) -> bool:
    """A catalog file of the current layout (it has the `live` column); opens only the
    ColumnVault, so a file of another layout is not touched."""
    if not os.path.exists(path):
        return False
    return "live" in {c[0] for c in kv.ColumnVault(path).list_columns()}


class CatalogVault:
    """Open `path` for reading, or for writing (`write`, the single writer; creates it)."""

    def __init__(self, path: str, write: bool = False):
        if not write and not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        self.write = write
        self.cols = kv.ColumnVault(path)
        self.kv = kv.KVault(path)
        if write:
            for name in FIXED:
                self.cols.ensure(name, "i64")
            self.cols.ensure("weight32", "bytes:4")
            self.cols.ensure("item", "str:utf8")
        raw = self.kv.get(GROUPS)
        self.groups: list[str] = [] if raw is None else PACK.unpack(raw, 0)
        self._group_ids = {g: i for i, g in enumerate(self.groups)}

    def info(self) -> dict:
        """The committed schema, generation, row count and source kind."""
        raw = self.kv.get(INFO)
        if raw is None:
            return {"schema": SCHEMA, "generation": 0, "rows": 0}
        return PACK.unpack(raw, 0)

    def rows(self) -> int:
        """The committed row count."""
        return int(self.info()["rows"])

    def read_fixed(
        self, n: int | None = None, names: tuple[str, ...] = FIXED
    ) -> dict[str, np.ndarray]:
        """The first `n` (default: committed) rows of the fixed columns `names` and
        the weights, one read each."""
        n = self.rows() if n is None else n
        out = {
            name: np.frombuffer(_raw(self.cols[name], n), dtype="<i8").astype(
                np.int64, copy=False
            )
            for name in names
        }
        weight = np.frombuffer(_raw(self.cols["weight32"], n), dtype="<f4")
        out["weight"] = weight.astype(np.float32, copy=False)
        return out

    def read_items(self, n: int | None = None) -> list[str]:
        """The scan item key of each of the first `n` (default: committed) rows."""
        n = self.rows() if n is None else n
        return list(self.cols["item"][0:n]) if n else []

    def fetch(self, i: int) -> dict:
        """The variable fields of row `i`."""
        return PACK.unpack(self.kv.get(row_key(i)), 0)

    def _group(self, name: str) -> int:
        gid = self._group_ids.get(name)
        if gid is None:
            gid = len(self.groups)
            self.groups.append(name)
            self._group_ids[name] = gid
        return gid

    def _fixed(self, row: CatalogRow, stamp: tuple, live: int) -> tuple:
        head = (row.w, row.h, int(row.has_alpha), live, self._group(row.group))
        return (*head, *text_chars(row), *stamp)

    def append(
        self, rows: list[CatalogRow], items: list[str], stamps: list[tuple]
    ) -> int:
        """Append rows (committed in one chunk); returns the first new id."""
        start = len(self.cols["item"])
        values = [self._fixed(r, s, 1) for r, s in zip(rows, stamps)]
        for k, name in enumerate(FIXED):
            self.cols[name].extend([v[k] for v in values])
        self.cols["weight32"].extend([WEIGHT.pack(r.weight) for r in rows])
        self.cols["item"].extend(list(items))
        with self.kv.cache():
            for i, row in enumerate(rows):
                self.kv[row_key(start + i)] = PACK.pack(var_fields(row))
        return start

    def update(self, i: int, row: CatalogRow, item: str, stamp: tuple) -> None:
        """Rewrite row `i` in place (its id stays)."""
        for name, value in zip(FIXED, self._fixed(row, stamp, 1)):
            self.cols[name][i] = value
        self.cols["weight32"][i] = WEIGHT.pack(row.weight)
        self.cols["item"][i] = item
        self.kv[row_key(i)] = PACK.pack(var_fields(row))

    def tombstone(self, i: int) -> None:
        """Row `i` is gone: live 0, weight 0 (its id is never reused for another item)."""
        self.cols["live"][i] = 0
        self.cols["weight32"][i] = WEIGHT.pack(0.0)

    def commit(self, generation: int, kind: str = "") -> dict:
        """Record the row count and generation, then checkpoint; returns the info."""
        info = {
            "schema": SCHEMA,
            "generation": generation,
            "rows": len(self.cols["item"]),
            "kind": kind,
        }
        self.kv[GROUPS] = PACK.pack(self.groups)
        self.kv[INFO] = PACK.pack(info)
        self.cols.checkpoint()
        self.kv.checkpoint()
        return info
