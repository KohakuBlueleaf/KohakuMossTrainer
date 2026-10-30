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
from kohakumoss_data.images import KcIndex, KcStore
from kohakumoss_data.registry import SOURCES
from kohakumoss_data.scan.walk import WalkItem, rows_of
from kohakumoss_data.types import CatalogRow

try:
    from kohakugbc_data.dataset import GbcDataset, read_info
except ImportError:
    GbcDataset = read_info = None

SHARD_FILES = ("manifest.parquet", "graphs.db", "meta.db", "images.zip")
PROBE_ROWS = 4096  # samples per walk item
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


def probe_shard(
    root: str,
    name: str,
    images: dict | None = None,
    start: int = 0,
    stop: int | None = None,
) -> list[CatalogRow]:
    """Samples [start, stop) of shard `name` as rows (stop None: to the end): size from
    the manifest, the prompt bound from its graph and metadata (`extra["prompt_chars"]`);
    with `images` (a KohakuClip image store, see GbcSource) the size is the store's
    image's and a sample the store lacks is left out."""
    ds = GbcDataset(root, shards=[name], image=None)
    sizes = ds.manifest.column("image_bytes").to_pylist()
    index = KcIndex(images["index"]) if images else None
    rows = []
    for i in range(start, len(ds) if stop is None else min(stop, len(ds))):
        s = ds.sample(i)
        w, h, stamp = int(s.w), int(s.h), f"gbc:{s.key}:{s.codec}:{sizes[i]}"
        if index is not None:
            row_at = index.find(int(s.key))
            if row_at < 0:
                continue
            w, h = index.size_of(row_at)
            stamp = f"kc:{s.key}:{int(index.cols['size'][row_at])}"
        row = CatalogRow(id=0, locator=f"{name}/{s.key}", w=w, h=h)
        row.extra["hash"] = stamp
        row.extra[PROMPT_CHARS] = prompt_bound(s.graph, s.meta or {}, w, h)
        rows.append(row)
    ds.close()
    return rows


@SOURCES.register("gbc")
class GbcSource:
    """A GBC dataset at `path` (`shards`: only these shard names). Rows carry no caption
    fields: the `gbc` caption template reads each sample's record. `images`
    ({"index": a kohakumoss_data.images index directory, "root": the shard directory or
    http(s) URL}) reads every image by its key from those KohakuClip shards instead of
    the dataset's own `images.zip` (rows the index lacks are left out)."""

    def __init__(
        self, path: str, shards: list[str] | None = None, images: dict | None = None
    ):
        if GbcDataset is None:
            raise ImportError(NEEDS)
        self.path = path
        self.shards = list(shards) if shards else None
        self.images = dict(images) if images else None
        self.store = KcStore(images["index"], images["root"]) if images else None
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
        images = entry.get("images")
        if images is None:
            return []
        if not isinstance(images, dict) or set(images) != {"index", "root"}:
            return [
                f"gbc source {path!r}: images must be {{'index': ..., 'root': ...}}"
            ]
        if not os.path.isfile(os.path.join(images["index"], "key.npy")):
            return [f"gbc source {path!r}: no image index at {images['index']!r}"]
        return []

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_ds"] = None
        return state

    def _dataset(self):
        """The dataset reader, opened once per process (its files open on first use)."""
        if self._ds is None:
            image = None if self.store is not None else "bytes"
            self._ds = GbcDataset(self.path, shards=self.shards, image=image)
        return self._ds

    def shard_samples(self) -> list[tuple[str, int]]:
        """(name, sample count) of every shard this source reads, in dataset order."""
        shards = read_info(self.path)["shards"]
        keep = [s for s in shards if self.shards is None or s["name"] in self.shards]
        return [(s["name"], int(s["samples"])) for s in keep]

    def shard_names(self) -> list[str]:
        return [name for name, _ in self.shard_samples()]

    def walk(self) -> list[WalkItem]:
        """One walk item per PROBE_ROWS samples of a shard (a scan probes them in
        parallel), keyed by shard and first sample."""
        store = f":{self.images['index']}" if self.images else ""
        root = os.path.abspath(self.path)
        items = []
        for name, samples in self.shard_samples():
            stamp = shard_stamp(self.path, name)
            for start in range(0, max(samples, 1), PROBE_ROWS):
                stop = start + PROBE_ROWS
                probe = partial(probe_shard, self.path, name, self.images, start, stop)
                items.append(
                    WalkItem(f"gbc:{root}:{name}:{start}{store}", stamp, probe)
                )
        return items

    def scan(self) -> Iterator[CatalogRow]:
        return rows_of(self.walk())

    def open(self, locator: str) -> bytes:
        key = locator.partition("/")[2]
        if self.store is not None:
            return self.store.read(int(key))
        return self._dataset().image(key)

    def record(self, locator: str) -> GbcRecord:
        """The graph and metadata of the row at `locator`."""
        ds, key = self._dataset(), locator.partition("/")[2]
        return GbcRecord(ds.graph(key), ds.meta_of(key))
