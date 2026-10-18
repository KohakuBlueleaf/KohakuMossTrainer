"""Plain folder tree: every image with `.txt` / `.caption` / `.json` sidecars."""

import os
from collections.abc import Iterator
from functools import partial

from kohakumoss_data.registry import SOURCES
from kohakumoss_data.scan.walk import WalkItem, file_stamp, listing, rows_of
from kohakumoss_data.sources.base import is_image, probe_file, read_bytes
from kohakumoss_data.types import CatalogRow


@SOURCES.register("folder")
class FolderSource:
    """All images under `path` (recursive by default); group = relative directory."""

    def __init__(
        self,
        path: str,
        recursive: bool = True,
        txt_as: str = "tags",
        hash_content: bool = False,
    ):
        self.path = path
        self.recursive = recursive
        self.txt_as = txt_as
        self.hash_content = hash_content

    def _items(self, rel: str, out: list[WalkItem]) -> None:
        """Files of `rel` (sorted), then its subdirectories depth first (os.walk's
        top-down order; symlinked directories are not entered)."""
        folder = os.path.join(self.path, rel) if rel else self.path
        entries = listing(folder)
        for name in entries.names:
            if name in entries.dirs and self.recursive or not is_image(name):
                continue
            locator = f"{rel}/{name}" if rel else name
            full = os.path.join(folder, name)
            probe = partial(
                probe_file, full, locator, rel, self.hash_content, self.txt_as
            )
            out.append(WalkItem(locator, file_stamp(entries, name), probe))
        if not self.recursive:
            return
        for name in entries.names:
            if name in entries.dirs and name not in entries.links:
                self._items(f"{rel}/{name}" if rel else name, out)

    def walk(self) -> list[WalkItem]:
        items: list[WalkItem] = []
        self._items("", items)
        return items

    def scan(self) -> Iterator[CatalogRow]:
        return rows_of(self.walk())

    def open(self, locator: str) -> bytes:
        return read_bytes(os.path.join(self.path, locator))
