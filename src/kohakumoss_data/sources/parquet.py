"""Parquet manifest: rows with an image path or image bytes column plus metadata."""

import os
from collections.abc import Iterator

from kohakumoss_data.format.header import probe_image
from kohakumoss_data.registry import SOURCES
from kohakumoss_data.seeds import content_hash
from kohakumoss_data.sources.base import apply_json, image_row, read_bytes
from kohakumoss_data.types import CatalogRow

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


@SOURCES.register("parquet")
class ParquetSource:
    """`image_col` holds relative paths, or `bytes_col` holds encoded images."""

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
        if self._table is None:
            self._table = pq.read_table(self.path, memory_map=True)
        return self._table

    def scan(self) -> Iterator[CatalogRow]:
        table = self._rows()
        records = table.to_pylist()
        for i, meta in enumerate(records):
            weight = float(meta.pop("weight", 1.0) or 1.0)
            group = str(meta.pop("group", "") or "")
            if self.bytes_col:
                data = meta.pop(self.bytes_col)
                if self.image_col:
                    meta.pop(self.image_col, None)
                w, h, alpha = probe_image(data)
                row = CatalogRow(
                    id=0, locator=f"row:{i}", w=w, h=h, has_alpha=alpha, group=group
                )
                row.extra["hash"] = content_hash(data)
            else:
                rel = str(meta.pop(self.image_col)).replace("\\", "/")
                full = os.path.join(self.root, rel)
                row = image_row(full, rel, group, self.hash_content)
            row.weight = weight
            apply_json(row, {k: v for k, v in meta.items() if v is not None})
            yield row

    def open(self, locator: str) -> bytes:
        if locator.startswith("row:"):
            index = int(locator[4:])
            return self._rows().column(self.bytes_col)[index].as_py()
        return read_bytes(os.path.join(self.root, locator))
