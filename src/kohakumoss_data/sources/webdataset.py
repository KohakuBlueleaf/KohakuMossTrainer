"""WebDataset tar shards: members grouped by key (`<key>.<ext>`), read by byte offset."""

import glob
import io
import json
import os
import tarfile
from collections.abc import Iterator

from kohakumoss_data.caption.tags import split_tags
from kohakumoss_data.format.header import probe_image
from kohakumoss_data.registry import SOURCES
from kohakumoss_data.seeds import content_hash
from kohakumoss_data.sources.base import IMAGE_EXTS, apply_json
from kohakumoss_data.types import CatalogRow


def _split(name: str) -> tuple[str, str]:
    base = os.path.basename(name)
    key, _, ext = base.partition(".")
    return os.path.join(os.path.dirname(name), key), "." + ext.lower()


def pick_image(members: dict[str, tarfile.TarInfo]):
    """The sample's image member: a plain image extension (`.png`) first, else the
    first (sorted) dotted one whose last suffix is an image (`.0.jpg`, `.seg.png`)."""
    plain = [e for e in members if e in IMAGE_EXTS]
    dotted = sorted(e for e in members if e.endswith(IMAGE_EXTS))
    ext = (sorted(plain) or dotted or [None])[0]
    return None if ext is None else members[ext]


@SOURCES.register("webdataset")
class WebDatasetSource:
    """`path` is a glob of .tar files (uncompressed); locator `tar_index:offset:size`.
    `skipped` lists the keys of the last scan that had no image member."""

    def __init__(self, path: str, txt_as: str = "tags"):
        self.path = path
        self.tars = sorted(glob.glob(path))
        self.txt_as = txt_as
        self.skipped: list[str] = []
        self._handles: dict[int, object] = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def scan(self) -> Iterator[CatalogRow]:
        self.skipped = []
        for t, tar_path in enumerate(self.tars):
            with tarfile.open(tar_path, "r:") as tar:
                groups: dict[str, dict[str, tarfile.TarInfo]] = {}
                for member in tar:
                    if member.isfile():
                        key, ext = _split(member.name)
                        groups.setdefault(key, {})[ext] = member
                for key in sorted(groups):
                    yield from self._row(tar, t, key, groups[key])
        if self.skipped:
            print(
                f"webdataset: {len(self.skipped)} keys without an image member "
                f"skipped (first: {self.skipped[0]})"
            )

    def _row(self, tar, t, key, members) -> Iterator[CatalogRow]:
        image = pick_image(members)
        if image is None:
            self.skipped.append(f"{os.path.basename(self.tars[t])}:{key}")
            return
        data = tar.extractfile(image).read()
        w, h, alpha = probe_image(data)
        locator = f"{t}:{image.offset_data}:{image.size}"
        row = CatalogRow(id=0, locator=locator, w=w, h=h, has_alpha=alpha)
        row.group = os.path.basename(self.tars[t])
        row.extra["hash"] = content_hash(data)
        row.extra["key"] = key
        if ".txt" in members:
            text = tar.extractfile(members[".txt"]).read().decode("utf-8").strip()
            if self.txt_as == "tags":
                row.tags = split_tags(text)
            else:
                row.caption_nl = text
        if ".json" in members:
            apply_json(
                row, json.load(io.BytesIO(tar.extractfile(members[".json"]).read()))
            )
        yield row

    def open(self, locator: str) -> bytes:
        t, offset, size = (int(p) for p in locator.split(":"))
        handle = self._handles.get(t)
        if handle is None:
            handle = open(self.tars[t], "rb")
            self._handles[t] = handle
        handle.seek(offset)
        return handle.read(size)
