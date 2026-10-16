"""JSON-lines manifest: one record per line with an image path and metadata."""

import json
import os
from collections.abc import Iterator

from kohakumoss_data.registry import SOURCES
from kohakumoss_data.sources.base import apply_json, image_row, read_bytes
from kohakumoss_data.types import CatalogRow

IMAGE_KEYS = ("image", "file", "path", "file_name")


@SOURCES.register("jsonl")
class JsonlSource:
    """`path` is the .jsonl file; image paths are relative to `root` (default: its dir)."""

    def __init__(self, path: str, root: str | None = None, hash_content: bool = False):
        self.path = path
        self.root = root or os.path.dirname(os.path.abspath(path))
        self.hash_content = hash_content

    def scan(self) -> Iterator[CatalogRow]:
        with open(self.path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                meta = json.loads(line)
                key = next((k for k in IMAGE_KEYS if k in meta), None)
                if key is None:
                    raise ValueError(f"{self.path}:{line_no + 1}: no image key")
                locator = str(meta.pop(key)).replace("\\", "/")
                weight = float(meta.pop("weight", 1.0))
                group = str(meta.pop("group", ""))
                full = os.path.join(self.root, locator)
                row = image_row(full, locator, group, self.hash_content)
                row.weight = weight
                apply_json(row, meta)
                yield row

    def open(self, locator: str) -> bytes:
        return read_bytes(os.path.join(self.root, locator))
