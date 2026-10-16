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


@SOURCES.register("webdataset")
class WebDatasetSource:
    """`path` is a glob of .tar files (uncompressed); locator `tar_index:offset:size`."""

    def __init__(self, path: str, txt_as: str = "tags"):
        self.path = path
        self.tars = sorted(glob.glob(path))
        self.txt_as = txt_as
        self._handles: dict[int, object] = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def scan(self) -> Iterator[CatalogRow]:
        for t, tar_path in enumerate(self.tars):
            with tarfile.open(tar_path, "r:") as tar:
                groups: dict[str, dict[str, tarfile.TarInfo]] = {}
                for member in tar:
                    if member.isfile():
                        key, ext = _split(member.name)
                        groups.setdefault(key, {})[ext] = member
                for key in sorted(groups):
                    yield from self._row(tar, t, key, groups[key])

    def _row(self, tar, t, key, members) -> Iterator[CatalogRow]:
        image = next((m for e, m in members.items() if e in IMAGE_EXTS), None)
        if image is None:
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
