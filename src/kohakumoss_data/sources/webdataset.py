"""WebDataset tar shards: members grouped by key (`<key>.<ext>`), read by byte offset."""

import glob
import json
import os
import posixpath
import tarfile
from collections.abc import Iterator
from functools import partial

from kohakumoss_common.log import get_logger
from kohakumoss_data.caption.tags import split_tags
from kohakumoss_data.format.header import probe_image
from kohakumoss_data.registry import SOURCES
from kohakumoss_data.scan.walk import WalkItem
from kohakumoss_data.seeds import content_hash
from kohakumoss_data.sources.base import IMAGE_EXTS, apply_json
from kohakumoss_data.types import CatalogRow

log = get_logger(__name__)


def _split(name: str) -> tuple[str, str]:
    """(sample key, lower-case extension) of a tar member name."""
    folder, base = posixpath.split(name)
    key, _, ext = base.partition(".")
    return posixpath.join(folder, key), "." + ext.lower()


def pick_image(members: dict[str, tarfile.TarInfo]):
    """The sample's image member: a plain image extension (`.png`) first, else the
    first (sorted) dotted one whose last suffix is an image (`.0.jpg`, `.seg.png`)."""
    plain = [e for e in members if e in IMAGE_EXTS]
    dotted = sorted(e for e in members if e.endswith(IMAGE_EXTS))
    ext = (sorted(plain) or dotted or [None])[0]
    return None if ext is None else members[ext]


def _row(tar, t: int, name: str, key: str, members, txt_as: str) -> CatalogRow:
    image = pick_image(members)
    data = tar.extractfile(image).read()
    w, h, alpha = probe_image(data)
    locator = f"{t}:{image.offset_data}:{image.size}"
    row = CatalogRow(id=0, locator=locator, w=w, h=h, has_alpha=alpha)
    row.group = name
    row.extra["hash"] = content_hash(data)
    row.extra["key"] = key
    if ".txt" in members:
        text = tar.extractfile(members[".txt"]).read().decode("utf-8").strip()
        if txt_as == "tags":
            row.tags = split_tags(text)
        else:
            row.caption_nl = text
    if ".json" in members:
        apply_json(row, json.loads(tar.extractfile(members[".json"]).read()))
    return row


def tar_rows(path: str, t: int, txt_as: str, skipped: list[str]) -> list[CatalogRow]:
    """The rows of tar `t` (sorted keys); keys without an image member go to `skipped`."""
    name = os.path.basename(path)
    rows = []
    with tarfile.open(path, "r:") as tar:
        groups: dict[str, dict[str, tarfile.TarInfo]] = {}
        for member in tar:
            if member.isfile():
                key, ext = _split(member.name)
                groups.setdefault(key, {})[ext] = member
        for key in sorted(groups):
            if pick_image(groups[key]) is None:
                skipped.append(f"{name}:{key}")
                continue
            rows.append(_row(tar, t, name, key, groups[key], txt_as))
    return rows


def _warn(skipped: list[str]) -> None:
    if skipped:
        log.warning(
            f"webdataset: {len(skipped)} keys without an image member skipped "
            f"(first: {skipped[0]})"
        )


def probe_tar(path: str, t: int, txt_as: str) -> list[CatalogRow]:
    """One tar as a walk item's probe."""
    skipped: list[str] = []
    rows = tar_rows(path, t, txt_as, skipped)
    _warn(skipped)
    return rows


@SOURCES.register("webdataset")
class WebDatasetSource:
    """`path` is a glob of .tar files (uncompressed); locator `tar_index:offset:size`.
    A walk item per tar, stamped with its index and stat. `skipped` lists the keys of
    the last `scan()` that had no image member."""

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

    def walk(self) -> list[WalkItem]:
        items = []
        for t, tar_path in enumerate(self.tars):
            st = os.stat(tar_path)
            stamp = (t, st.st_size, st.st_mtime_ns, -1, -1, -1, -1, -1)
            probe = partial(probe_tar, tar_path, t, self.txt_as)
            items.append(WalkItem(f"tar:{os.path.abspath(tar_path)}", stamp, probe))
        return items

    def scan(self) -> Iterator[CatalogRow]:
        self.skipped = []
        for t, tar_path in enumerate(self.tars):
            yield from tar_rows(tar_path, t, self.txt_as, self.skipped)
        _warn(self.skipped)

    def open(self, locator: str) -> bytes:
        t, offset, size = (int(p) for p in locator.split(":"))
        handle = self._handles.get(t)
        if handle is None:
            handle = open(self.tars[t], "rb")
            self._handles[t] = handle
        handle.seek(offset)
        return handle.read(size)
