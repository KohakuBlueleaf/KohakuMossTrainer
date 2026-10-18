"""Catalog-file maintenance: dropping a file with its lock / marker, and compacting it
(rewritten dense in walk order, unchanged rows reused)."""

import gc
import glob
import os

from kohakumoss_data.catalog.table import open_catalog
from kohakumoss_data.catalog.vault import STAMPS, CatalogVault
from kohakumoss_data.format.catalog import remove_vault
from kohakumoss_data.scan import lock
from kohakumoss_data.scan.runner import Appender, probe_all

SIDE = (lock.LOCK, lock.READY)


def drop(path: str) -> bool:
    """Remove a catalog file with its lock / marker; refuses while a writer holds it."""
    if lock.held(path):
        raise RuntimeError(f"{path} is being written (lock held); stop that run first")
    existed = os.path.exists(path)
    remove_vault(path)
    for suffix in SIDE:
        if os.path.exists(path + suffix):
            os.remove(path + suffix)
    return existed


def drop_all(cache_dir: str) -> list[str]:
    """Remove every catalog file under `<cache_dir>/catalog`; the removed paths."""
    paths = sorted(glob.glob(os.path.join(cache_dir, "catalog", "*.db")))
    return [p for p in paths if drop(p)]


def compact(source, path: str, workers: int = 8) -> int:
    """Rewrite `path` with only the live rows of a fresh walk, in walk order (new ids,
    what a cold scan writes); rows whose stamps are unchanged are copied, not probed.
    Returns the row count."""
    if not lock.acquire(path):
        raise RuntimeError(f"{path} is being written (lock held)")
    try:
        old = open_catalog(path)
        old_vault = CatalogVault(path)
        items = old_vault.read_items(len(old))
        fixed = old_vault.read_fixed(len(old))
        by_key: dict[str, list[int]] = {}
        for i, key in enumerate(items):
            if old.live[i]:
                by_key.setdefault(key, []).append(i)
        walked = source.walk()
        fresh = [
            w
            for w in walked
            if w.key not in by_key
            or tuple(fixed[s][by_key[w.key][0]] for s in STAMPS) != w.stamp
        ]
        probed = dict(zip((w.key for w in fresh), probe_all(fresh, workers)))
        tmp = path + ".compact"
        remove_vault(tmp)
        out = CatalogVault(tmp, write=True)
        prev = old_vault.info()
        appender = Appender(out, int(prev["generation"]) + 1, prev.get("kind", ""))
        count = 0
        for w in walked:
            rows = probed.get(w.key)
            if rows is None:
                rows = [old[i] for i in by_key[w.key]]
            appender.add(rows, w.key, w.stamp)
            count += len(rows)
        appender.flush()
        info = out.commit(appender.gen, appender.kind)
        # Every handle on both files is closed before the files are swapped.
        del out, old_vault, old, appender
        gc.collect()
        remove_vault(path)
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(tmp + suffix):
                os.replace(tmp + suffix, path + suffix)
        lock.write_ready(path, info)
        return count
    finally:
        lock.release(path)
