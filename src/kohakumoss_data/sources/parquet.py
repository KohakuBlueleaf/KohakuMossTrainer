"""Parquet manifest: rows with an image path or image bytes column plus metadata."""

import os
from collections.abc import Iterator
from functools import partial

from kohakumoss_data.format.header import probe_image
from kohakumoss_data.registry import SOURCES
from kohakumoss_data.scan.walk import Listings, WalkItem, rows_of
from kohakumoss_data.seeds import content_hash
from kohakumoss_data.sources.base import (
    Occurrences,
    apply_json,
    content_stamp,
    probe_manifest,
    read_bytes,
)
from kohakumoss_data.types import CatalogRow

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

BATCH_ROWS = 1024


def _take_meta(meta: dict) -> tuple[float, str, dict]:
    """(weight, group, the remaining non-null fields) of a record; weight 1 and group ""
    when absent or null."""
    weight = meta.pop("weight", None)
    group = meta.pop("group", None)
    rest = {k: v for k, v in meta.items() if v is not None}
    return 1.0 if weight is None else float(weight), group or "", rest


def probe_bytes_table(path: str, image_col: str | None, bytes_col: str) -> list:
    """Every row of a bytes-column parquet file (one walk item: the file), read in
    record batches of BATCH_ROWS."""
    rows = []
    batches = pq.ParquetFile(path, memory_map=True).iter_batches(BATCH_ROWS)
    for batch in batches:
        for meta in batch.to_pylist():
            data = meta.pop(bytes_col)
            if image_col:
                meta.pop(image_col, None)
            weight, group, rest = _take_meta(meta)
            w, h, alpha = probe_image(data)
            row = CatalogRow(
                id=0,
                locator=f"row:{len(rows)}",
                w=w,
                h=h,
                has_alpha=alpha,
                group=str(group),
                weight=weight,
            )
            row.extra["hash"] = content_hash(data)
            apply_json(row, rest)
            rows.append(row)
    return rows


@SOURCES.register("parquet")
class ParquetSource:
    """`image_col` holds relative paths (an item per row, keyed `path#n`, stamped with
    the image's stat and the row's content), or `bytes_col` holds encoded images (one
    item: the file, stamped with its stat)."""

    def __init__(
        self,
        path: str,
        image_col: str | None = "image",
        bytes_col: str | None = None,
        root: str | None = None,
        hash_content: bool = False,
    ):
        if pq is None:
            raise ImportError("the parquet source needs pyarrow")
        if not (image_col or bytes_col):
            raise ValueError("the parquet source needs image_col or bytes_col")
        self.path = path
        self.image_col = image_col
        self.bytes_col = bytes_col
        self.root = root or os.path.dirname(os.path.abspath(path))
        self.hash_content = hash_content
        self._table = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_table"] = None
        return state

    def _rows(self):
        """The table (only the bytes column in bytes mode), read once per process."""
        if self._table is None:
            columns = [self.bytes_col] if self.bytes_col else None
            self._table = pq.read_table(self.path, columns=columns, memory_map=True)
        return self._table

    def walk(self) -> list[WalkItem]:
        if self.bytes_col:
            st = os.stat(self.path)
            stamp = (st.st_size, st.st_mtime_ns, -1, -1, -1, -1, -1, -1)
            probe = partial(
                probe_bytes_table, self.path, self.image_col, self.bytes_col
            )
            return [WalkItem(f"file:{os.path.abspath(self.path)}", stamp, probe)]
        items, stats, keys = [], Listings(), Occurrences()
        for meta in self._rows().to_pylist():
            rel = str(meta.pop(self.image_col)).replace("\\", "/")
            weight, group, rest = _take_meta(meta)
            group = str(group)
            full = os.path.join(self.root, rel)
            content = repr(sorted(rest.items())) + f"|{weight}|{group}"
            stamp = (*stats.stat(full), content_stamp(content), 0, -1, -1, -1, -1)
            probe = partial(
                probe_manifest, full, rel, group, self.hash_content, weight, rest
            )
            items.append(WalkItem(keys.key(rel), stamp, probe))
        return items

    def scan(self) -> Iterator[CatalogRow]:
        return rows_of(self.walk())

    def open(self, locator: str) -> bytes:
        if locator.startswith("row:"):
            index = int(locator[4:])
            return self._rows().column(self.bytes_col)[index].as_py()
        return read_bytes(os.path.join(self.root, locator))
