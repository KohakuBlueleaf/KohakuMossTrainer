"""Kohya-style folders: `<repeats>_<name>/image + sidecars`; repeats become row weight."""

import os
import re
from collections.abc import Iterator
from functools import partial

from kohakumoss_data.registry import SOURCES
from kohakumoss_data.scan.walk import WalkItem, file_stamp, listing, rows_of
from kohakumoss_data.sources.base import is_image, probe_file, read_bytes
from kohakumoss_data.types import CatalogRow

PREFIX = re.compile(r"^((?:\d+_)+)(.*)$")


def parse_folder(name: str, mode: str) -> tuple[float, str]:
    """(repeats, name without numeric prefixes) of a folder name; `mode` `first` (the
    leading integer), `product` (of every leading integer) or `ignore` (1)."""
    match = PREFIX.match(name)
    if match is None:
        return 1.0, name
    numbers = [int(p) for p in match.group(1).strip("_").split("_")]
    rest = match.group(2)
    match mode:
        case "first":
            return float(numbers[0]), rest
        case "product":
            repeats = 1
            for n in numbers:
                repeats *= n
            return float(repeats), rest
        case "ignore":
            return 1.0, rest
        case _:
            raise ValueError(f"repeat_prefix must be first|product|ignore, got {mode}")


def artist_of(folder_name: str, pattern: str | None) -> str | None:
    """Artist from a folder remainder: `pattern` None, "name" (the whole remainder) or a
    regex whose first group (or whole match) is the artist."""
    match pattern:
        case None | "":
            return None
        case "name":
            return folder_name
        case _:
            found = re.search(pattern, folder_name)
            if found is None:
                return None
            return found.group(1) if found.groups() else found.group(0)


@SOURCES.register("kohya")
class KohyaSource:
    """Every image in the first-level subfolders of `path` (and, with `include_root`,
    the images in `path` itself, repeat 1); orphan sidecars are ignored."""

    def __init__(
        self,
        path: str,
        repeat_prefix: str = "first",
        artist_from_folder: str | None = None,
        txt_as: str = "tags",
        hash_content: bool = False,
        include_root: bool = False,
    ):
        self.path = path
        self.repeat_prefix = repeat_prefix
        self.artist_from_folder = artist_from_folder
        self.txt_as = txt_as
        self.hash_content = hash_content
        self.include_root = include_root

    def _images(self, entries, folder: str, repeats: float, artist) -> list[WalkItem]:
        """The image items of one listed folder (`folder` "" = the root)."""
        items = []
        folder_path = os.path.join(self.path, folder) if folder else self.path
        for file in entries.names:
            if file in entries.dirs or not is_image(file):
                continue
            locator = f"{folder}/{file}" if folder else file
            full = os.path.join(folder_path, file)
            probe = partial(
                probe_file,
                full,
                locator,
                folder,
                self.hash_content,
                self.txt_as,
                repeats,
                artist,
            )
            items.append(WalkItem(locator, file_stamp(entries, file), probe))
        return items

    def walk(self) -> list[WalkItem]:
        """One item per image in walk order (root images first when included, then
        folders and files sorted); one listing per folder gives every stamp. Raises
        ValueError when no image is found."""
        top = listing(self.path)
        items = self._images(top, "", 1.0, None) if self.include_root else []
        for folder in top.names:
            if folder not in top.dirs:
                continue
            repeats, name = parse_folder(folder, self.repeat_prefix)
            artist = artist_of(name, self.artist_from_folder)
            items += self._images(
                listing(os.path.join(self.path, folder)), folder, repeats, artist
            )
        if not items:
            raise ValueError(
                f"kohya source: no images in the subfolders of {self.path} (the layout "
                "is <repeats>_<name>/ folders of images); images in the folder itself "
                "need include_root=True, or use source 'folder'"
            )
        return items

    def scan(self) -> Iterator[CatalogRow]:
        return rows_of(self.walk())

    def open(self, locator: str) -> bytes:
        return read_bytes(os.path.join(self.path, locator))
