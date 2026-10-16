"""Plain folder tree: every image with `.txt` / `.caption` / `.json` sidecars."""

import os
from collections.abc import Iterator

from kohakumoss_data.registry import SOURCES
from kohakumoss_data.sources.base import apply_sidecars, image_row, is_image, read_bytes
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

    def _files(self) -> Iterator[str]:
        if not self.recursive:
            for file in sorted(os.listdir(self.path)):
                if is_image(file):
                    yield file
            return
        for root, dirs, files in os.walk(self.path):
            dirs.sort()
            rel = os.path.relpath(root, self.path)
            for file in sorted(files):
                if is_image(file):
                    yield file if rel == "." else f"{rel}/{file}".replace("\\", "/")

    def scan(self) -> Iterator[CatalogRow]:
        for locator in self._files():
            full = os.path.join(self.path, locator)
            group = os.path.dirname(locator)
            row = image_row(full, locator, group, self.hash_content)
            apply_sidecars(row, os.path.splitext(full)[0], self.txt_as)
            yield row

    def open(self, locator: str) -> bytes:
        return read_bytes(os.path.join(self.path, locator))
