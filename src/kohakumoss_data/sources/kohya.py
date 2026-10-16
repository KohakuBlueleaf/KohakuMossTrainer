"""Kohya-style folders: `<repeats>_<name>/image + sidecars`; repeats become row weight."""

import os
import re
from collections.abc import Iterator

from kohakumoss_data.registry import SOURCES
from kohakumoss_data.sources.base import apply_sidecars, image_row, is_image, read_bytes
from kohakumoss_data.types import CatalogRow

PREFIX = re.compile(r"^((?:\d+_)+)(.*)$")


def parse_folder(name: str, mode: str) -> tuple[float, str]:
    """(repeats, name without numeric prefixes) for a folder name under `mode`.

    `first`: the leading integer (kohya); `product`: product of every leading integer;
    `ignore`: 1.
    """
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


def artist_of(folder_name: str, rule: str | None) -> str | None:
    """Artist from a folder remainder: None, "name" (the whole remainder) or a regex
    whose first group (or whole match) is the artist."""
    match rule:
        case None | "":
            return None
        case "name":
            return folder_name
        case _:
            found = re.search(rule, folder_name)
            if found is None:
                return None
            return found.group(1) if found.groups() else found.group(0)


@SOURCES.register("kohya")
class KohyaSource:
    """Every image in the first-level subfolders of `path`; orphan sidecars are ignored."""

    def __init__(
        self,
        path: str,
        repeat_prefix: str = "first",
        artist_from_folder: str | None = None,
        txt_as: str = "tags",
        hash_content: bool = False,
    ):
        self.path = path
        self.repeat_prefix = repeat_prefix
        self.artist_from_folder = artist_from_folder
        self.txt_as = txt_as
        self.hash_content = hash_content

    def scan(self) -> Iterator[CatalogRow]:
        for folder in sorted(os.listdir(self.path)):
            folder_path = os.path.join(self.path, folder)
            if not os.path.isdir(folder_path):
                continue
            repeats, name = parse_folder(folder, self.repeat_prefix)
            artist = artist_of(name, self.artist_from_folder)
            for file in sorted(os.listdir(folder_path)):
                if not is_image(file):
                    continue
                locator = f"{folder}/{file}"
                full = os.path.join(folder_path, file)
                row = image_row(full, locator, folder, self.hash_content)
                row.weight = repeats
                row.artist = artist
                apply_sidecars(row, os.path.splitext(full)[0], self.txt_as)
                yield row

    def open(self, locator: str) -> bytes:
        return read_bytes(os.path.join(self.path, locator))
