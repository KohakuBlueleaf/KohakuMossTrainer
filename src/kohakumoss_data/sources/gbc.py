"""The GBC dataset format (KohakuGBC, `kohakugbc-data`): images, GBC graphs and a free-form
metadata dict per sample, in self-contained shards. One walk item per shard; a row's
locator is `<shard>/<key>`; `record(locator)` gives its graph and metadata. See
docs/concepts/data.md#gbc-datasets."""

import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial

from kohakumoss_data.catalog.text import PROMPT_CHARS
from kohakumoss_data.registry import SOURCES
from kohakumoss_data.scan.walk import WalkItem, rows_of
from kohakumoss_data.types import CatalogRow

try:
    from kohakugbc_data.dataset import GbcDataset, read_info
except ImportError:
    GbcDataset = read_info = None

SHARD_FILES = ("manifest.parquet", "graphs.db", "meta.db", "images.zip")
NEEDS = (
    "the gbc source needs kohakugbc-data (pip install 'kohakumoss[gbc]', Python 3.11+)"
)


@dataclass(frozen=True)
class GbcRecord:
    """One sample's GBC graph (`kohakugbc_data.GbcGraph`) and metadata dict."""

    graph: object
    meta: dict


def meta_chars(meta: dict) -> int:
    """Characters of every top-level string and string-list value of `meta` (list items
    joined by ", "): a bound on what a metadata formatter can write."""
    out = 0
    for value in meta.values():
        match value:
            case str():
                out += len(value) + 2
            case list() | tuple():
                out += sum(len(str(v)) + 2 for v in value)
    return out


def prompt_bound(graph, meta: dict, w: int, h: int) -> int:
    """A bound on any gbc caption of the sample: the graph's full-detail prompt text
    plus every metadata string."""
    return len(graph.prompt(width=w, height=h)) + meta_chars(meta)


def shard_stamp(root: str, name: str) -> tuple:
    """(size, mtime_ns) of each of the shard's four files (-1 when missing)."""
    out = []
    for file in SHARD_FILES:
        try:
            st = os.stat(os.path.join(root, "shards", name, file))
            out += [st.st_size, st.st_mtime_ns]
        except OSError:
            out += [-1, -1]
    return tuple(out)


def probe_shard(root: str, name: str) -> list[CatalogRow]:
    """Every sample of shard `name` as a row: size from the manifest, the prompt bound
    from its graph and metadata (`extra["prompt_chars"]`)."""
    ds = GbcDataset(root, shards=[name], image=None)
    sizes = ds.manifest.column("image_bytes").to_pylist()
    rows = []
    for i in range(len(ds)):
        s = ds.sample(i)
        row = CatalogRow(id=0, locator=f"{name}/{s.key}", w=int(s.w), h=int(s.h))
        row.extra["hash"] = f"gbc:{s.key}:{s.codec}:{sizes[i]}"
        row.extra[PROMPT_CHARS] = prompt_bound(s.graph, s.meta or {}, s.w, s.h)
        rows.append(row)
    ds.close()
    return rows


@SOURCES.register("gbc")
class GbcSource:
    """A GBC dataset at `path` (`shards`: only these shard names). Rows carry no caption
    fields: the `gbc` caption template reads each sample's record."""

    def __init__(self, path: str, shards: list[str] | None = None):
        if GbcDataset is None:
            raise ImportError(NEEDS)
        self.path = path
        self.shards = list(shards) if shards else None
        self._ds = None

    @staticmethod
    def problems(entry: dict) -> list[str]:
        """Knob problems of a gbc dataset entry."""
        if GbcDataset is None:
            return [
                NEEDS + (" (this is Python 3.10)" if sys.version_info < (3, 11) else "")
            ]
        path = str(entry.get("path", ""))
        if not os.path.isfile(os.path.join(path, "dataset.json")):
            return [f"gbc source {path!r}: no dataset.json (not a GBC dataset)"]
        return []

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_ds"] = None
        return state

    def _dataset(self):
        """The dataset reader, opened once per process (its files open on first use)."""
        if self._ds is None:
            self._ds = GbcDataset(self.path, shards=self.shards, image="bytes")
        return self._ds

    def shard_names(self) -> list[str]:
        names = [s["name"] for s in read_info(self.path)["shards"]]
        return [n for n in names if self.shards is None or n in self.shards]

    def walk(self) -> list[WalkItem]:
        return [
            WalkItem(
                f"gbc:{os.path.abspath(self.path)}:{name}",
                shard_stamp(self.path, name),
                partial(probe_shard, self.path, name),
            )
            for name in self.shard_names()
        ]

    def scan(self) -> Iterator[CatalogRow]:
        return rows_of(self.walk())

    def open(self, locator: str) -> bytes:
        return self._dataset().image(locator.partition("/")[2])

    def record(self, locator: str) -> GbcRecord:
        """The graph and metadata of the row at `locator`."""
        ds, key = self._dataset(), locator.partition("/")[2]
        return GbcRecord(ds.graph(key), ds.meta_of(key))
