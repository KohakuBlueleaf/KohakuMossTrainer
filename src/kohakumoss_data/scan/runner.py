"""The cached scan: a source's walk compared with its catalog file, re-probing only new
and changed items in worker processes; rescan modes False / True / "never".
See docs/concepts/data.md#the-catalog-cache."""

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial

import numpy as np

from kohakumoss_common.log import get_logger
from kohakumoss_data.catalog.table import Catalog, open_catalog
from kohakumoss_data.catalog.vault import (
    NO_STAMP,
    STAMPS,
    CatalogVault,
    is_catalog_vault,
)
from kohakumoss_data.scan import lock
from kohakumoss_data.scan.walk import WalkItem

log = get_logger(__name__)
RESCAN = (False, True, "never")
CHUNK = 4096
POOL_MIN = 64


@dataclass
class ScanCounts:
    """Rows of the scan: from the file, freshly probed, and the probe calls made."""

    cached: int = 0
    probed: int = 0
    probes: int = 0


def _call(probe):
    return probe()


def _scanned(source) -> list:
    return list(source.scan())


def items_of(source) -> list[WalkItem]:
    """`source.walk()`, or one item holding its whole `scan()` for a source without a
    walk (stamped as never changing: rescanned only with DATA_RESCAN=True)."""
    walk = getattr(source, "walk", None)
    if walk is None:
        return [WalkItem("scan", NO_STAMP, partial(_scanned, source))]
    return walk()


def probe_all(items: list[WalkItem], workers: int) -> list[list]:
    """Each item's rows, in item order; a process pool when there is enough to probe."""
    if workers <= 1 or len(items) < POOL_MIN:
        return [item.probe() for item in items]
    chunk = max(1, len(items) // (workers * 8))
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=spawn) as pool:
        return list(pool.map(_call, [i.probe for i in items], chunksize=chunk))


def _by_item(items: list[str], live: np.ndarray) -> dict[str, list[int]]:
    """Item key -> its live row ids, or its tombstoned ids when none is live."""
    every: dict[str, list[int]] = {}
    for i, key in enumerate(items):
        every.setdefault(key, []).append(i)
    return {key: [i for i in ids if live[i]] or ids for key, ids in every.items()}


class Appender:
    """Rows appended in chunks, each chunk committed (generation `gen`)."""

    def __init__(self, vault: CatalogVault, gen: int, kind: str):
        self.vault, self.gen, self.kind = vault, gen, kind
        self.rows, self.items, self.stamps = [], [], []

    def add(self, rows: list, key: str, stamp: tuple) -> None:
        for row in rows:
            self.rows.append(row)
            self.items.append(key)
            self.stamps.append(stamp)
        if len(self.rows) >= CHUNK:
            self.flush()

    def flush(self) -> None:
        if self.rows:
            self.vault.append(self.rows, self.items, self.stamps)
            self.vault.commit(self.gen, self.kind)
            self.rows, self.items, self.stamps = [], [], []


def update(source, path: str, rescan, workers: int, kind: str) -> ScanCounts:
    """Bring the catalog file at `path` up to date with `source` (the single writer);
    a file of another layout is deleted and rebuilt by a full scan."""
    if os.path.exists(path) and not is_catalog_vault(path):
        log.warning(f"catalog {path} has another layout: rebuilt by a full scan")
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(path + suffix):
                os.remove(path + suffix)
    vault = CatalogVault(path, write=True)
    info = vault.info()
    n = int(info["rows"])
    counts = ScanCounts()
    fixed = vault.read_fixed(n)
    live = fixed["live"].astype(bool)
    if rescan == "never" and n:
        counts.cached = int(live.sum())
        return counts
    stamps = np.stack([fixed[name] for name in STAMPS], 1)
    by_item = _by_item(vault.read_items(n), live)
    walked = items_of(source)
    seen, new, changed = set(), [], []
    for item in walked:
        seen.add(item.key)
        ids = by_item.get(item.key)
        if ids is None:
            new.append(item)
        elif rescan is True or not live[ids[0]] or tuple(stamps[ids[0]]) != item.stamp:
            changed.append((item, ids))
        else:
            counts.cached += len(ids)
    gen = int(info["generation"]) + 1
    results = probe_all([i for i, _ in changed] + new, workers)
    counts.probes = len(results)
    appender = Appender(vault, gen, kind)
    for (item, ids), rows in zip(changed, results[: len(changed)]):
        counts.probed += len(rows)
        if len(rows) == len(ids):
            for i, row in zip(ids, rows):
                vault.update(i, row, item.key, item.stamp)
            continue
        for i in ids:
            vault.tombstone(i)
        appender.add(rows, item.key, item.stamp)
    for key, ids in by_item.items():
        if key not in seen:
            for i in ids:
                if live[i]:
                    vault.tombstone(i)
    for item, rows in zip(new, results[len(changed) :]):
        counts.probed += len(rows)
        appender.add(rows, item.key, item.stamp)
    appender.flush()
    lock.write_ready(path, vault.commit(gen, kind))
    return counts


def scan_catalog(
    source,
    path: str | None,
    writer: bool = True,
    rescan=False,
    workers: int = 8,
    wait: float = 3600.0,
    kind: str = "",
) -> Catalog:
    """The catalog of `source` through its catalog file at `path` (None: no cache).
    The writer takes the file's lock and updates it; a reader (or a writer that finds
    the lock held by a live process) waits for the completion marker and reads."""
    if rescan not in RESCAN:
        raise ValueError(f"DATA_RESCAN must be one of {RESCAN}, got {rescan!r}")
    start = time.perf_counter()
    counts = ScanCounts()
    if path is None:
        rows = [r for rs in probe_all(items_of(source), workers) for r in rs]
        counts.probed = counts.probes = len(rows)
        catalog = Catalog(rows)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if writer and lock.acquire(path):
            try:
                lock.clear_ready(path)
                counts = update(source, path, rescan, workers, kind)
                if not lock.read_ready(path):
                    lock.write_ready(path, CatalogVault(path).info())
            finally:
                lock.release(path)
        else:
            lock.wait_ready(path, wait)
        catalog = open_catalog(path)
        counts.cached = counts.cached or int(catalog.live.sum()) - counts.probed
    seconds = time.perf_counter() - start
    log.info(
        f"catalog: {int(catalog.live.sum())} images ({counts.cached} from cache, "
        f"{counts.probed} probed) in {seconds:.2f} s"
    )
    catalog.counts = counts
    return catalog
