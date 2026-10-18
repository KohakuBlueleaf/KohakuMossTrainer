"""Bench the catalog: scan a synthetic file tree (the uncached serial scan, a cold
cached scan, a warm one), open a large catalog (write, warm open, RSS, per-row fetch;
against the per-field column layout, read eagerly), and the token-budget fill of
TOKEN_BUDGET estimate vs render on a dataset.

    python scripts/bench/data/catalog.py tree --root /tmp/tree --files 20000
    python scripts/bench/data/catalog.py catalog --root /tmp/cat --rows 1000000
    python scripts/bench/data/catalog.py budget --dataset <moss_shard dataset dir>

Every result is one JSON line (and `--out`).
"""

import argparse
import io
import json
import os
import random
import subprocess
import sys
import time

import kohakuvault as kv
import numpy as np
from PIL import Image

from kohakumoss_data.config import DataConfig
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.format.catalog import read_catalog, write_catalog
from kohakumoss_data.registry import SOURCES
from kohakumoss_data.scan.manage import drop
from kohakumoss_data.scan.paths import scan_path
from kohakumoss_data.scan.runner import scan_catalog
from kohakumoss_data.stats import plan_stats
from kohakumoss_data.types import CatalogRow


def png_bytes(w: int = 16, h: int = 16) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def rss_mb() -> float:
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        return float("nan")


def make_tree(root: str, files: int, per_folder: int = 1000) -> None:
    if os.path.exists(os.path.join(root, ".done")):
        return
    png = png_bytes()
    for i in range(files):
        folder = os.path.join(root, f"2_f{i // per_folder:04d}")
        if i % per_folder == 0:
            os.makedirs(folder, exist_ok=True)
        stem = os.path.join(folder, f"{i:08d}")
        with open(stem + ".png", "wb") as fh:
            fh.write(png)
        with open(stem + ".txt", "w", encoding="utf-8") as fh:
            fh.write(f"tag_a, tag_b, artist_{i % 97}, t{i}")
    open(os.path.join(root, ".done"), "w").close()


def bench_tree(args) -> dict:
    make_tree(args.root, args.files)
    source = SOURCES.get("kohya")(args.root)
    cache = os.path.join(args.root + "-cache")
    path = scan_path(cache, {"source": "kohya", "path": args.root})
    drop(path)
    t = time.perf_counter()
    plain = list(source.scan())
    serial_s = time.perf_counter() - t
    t = time.perf_counter()
    cold = scan_catalog(source, path, workers=args.workers)
    cold_s = time.perf_counter() - t
    t = time.perf_counter()
    warm = scan_catalog(source, path, workers=args.workers)
    warm_s = time.perf_counter() - t
    same = len(plain) == len(warm) and all(
        plain[i] == warm[i] for i in random.Random(0).sample(range(len(plain)), 200)
    )
    return {
        "bench": "tree",
        "files": args.files,
        "workers": args.workers,
        "uncached_serial_s": serial_s,
        "cold_cached_s": cold_s,
        "warm_s": warm_s,
        "warm_probed": warm.counts.probed,
        "cold_probed": cold.counts.probed,
        "same_rows": bool(same),
    }


def synthetic_rows(n: int):
    for i in range(n):
        yield CatalogRow(
            id=0,
            locator=f"f{i // 1000:05d}/{i:09d}.png",
            w=512 + i % 509,
            h=512 + i % 311,
            group=f"g{i % 997}",
            weight=float(1 + i % 3),
            tags=["1girl", "solo", "long_hair", f"artist_{i % 97}", f"t{i}"],
            caption_nl="a synthetic caption" if i % 2 else None,
            extra={"hash": f"stat:{i}:{i}"},
        )


PER_FIELD = {
    "locator": "str:utf8", "w": "i64", "h": "i64", "has_alpha": "i64",
    "group": "str:utf8", "weight32": "bytes:4", "tags": "msgpack",
    "caption_nl": "msgpack", "artist": "msgpack", "rating": "msgpack",
    "json": "msgpack", "extra": "msgpack",
}  # fmt: skip


def write_per_field(path: str, n: int) -> None:
    """A catalog in the per-field column layout (one ColumnVault column per field)."""
    vault = kv.ColumnVault(path)
    cols = {name: vault.ensure(name, dtype) for name, dtype in PER_FIELD.items()}
    chunk = []
    for row in synthetic_rows(n):
        chunk.append(row)
        if len(chunk) == 65536:
            _per_field_chunk(cols, chunk)
            chunk = []
    _per_field_chunk(cols, chunk)
    vault.checkpoint()


def _per_field_chunk(cols, rows) -> None:
    if not rows:
        return
    for name in PER_FIELD:
        if name == "weight32":
            cols[name].extend([np.float32(r.weight).tobytes() for r in rows])
        elif name == "has_alpha":
            cols[name].extend([int(r.has_alpha) for r in rows])
        else:
            cols[name].extend([getattr(r, name) for r in rows])


def child_open(path: str) -> dict:
    """Open in a fresh process: time, RSS delta, per-row fetch latency."""
    base = rss_mb()
    t = time.perf_counter()
    cat = read_catalog(path)
    open_s = time.perf_counter() - t
    after = rss_mb()
    ids = np.random.default_rng(0).integers(0, len(cat), 2000).tolist()
    t = time.perf_counter()
    for i in ids:
        cat[i]
    fetch_us = (time.perf_counter() - t) / len(ids) * 1e6
    return {
        "rows": len(cat),
        "open_s": open_s,
        "rss_mb": after - base,
        "fetch_us": fetch_us,
    }


def bench_catalog(args) -> dict:
    os.makedirs(args.root, exist_ok=True)
    current = os.path.join(args.root, f"catalog-{args.rows}.db")
    per_field = os.path.join(args.root, f"per_field-{args.rows}.db")
    out = {"bench": "catalog", "rows": args.rows}
    if not os.path.exists(current):
        t = time.perf_counter()
        write_catalog(current, synthetic_rows(args.rows))
        out["write_catalog_s"] = time.perf_counter() - t
    if not args.skip_per_field and not os.path.exists(per_field):
        t = time.perf_counter()
        write_per_field(per_field, args.rows)
        out["write_per_field_s"] = time.perf_counter() - t
    out["file_mb"] = os.path.getsize(current) / 2**20
    for tag, path in (("catalog", current), ("per_field", per_field)):
        if tag == "per_field" and args.skip_per_field:
            continue
        cmd = [sys.executable, __file__, "child", "--path", path]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out[tag] = json.loads(res.stdout.strip().splitlines()[-1])
    return out


def bench_budget(args) -> dict:
    out = {"bench": "budget", "dataset": os.path.basename(args.dataset.rstrip("/"))}
    for mode in ("estimate", "render"):
        cfg = DataConfig(
            datasets=[{"source": args.source, "path": args.dataset}],
            latent_cache="off",
            text_cache="off",
            token_budget=mode,
            max_tokens=args.max_tokens,
            target_area=args.area,
            caption={"template": "tags", "keep": ["artist"], "shuffle": True},
        )
        t = time.perf_counter()
        ds = MossDataset(cfg)
        stats = plan_stats(ds, 0, args.max_tokens)
        render_lens = []
        planner = ds.planner
        for batch in planner.epoch(0):
            txt = sum(
                ds.estimate_text_tokens(ds.caption(it.row_id, it.caption_variant))
                for it in batch
            )
            img = sum(planner.sizes(it)[0] for it in batch)
            render_lens.append((img + txt) / args.max_tokens)
        out[mode] = {
            "batches": stats["batches"],
            "samples_per_batch": stats["samples_per_batch"],
            "planned_fill": stats["budget_fill"],
            "rendered_fill": float(np.mean(render_lens)),
            "rendered_over_budget": float(np.mean(np.array(render_lens) > 1.0)),
            "plan_s": time.perf_counter() - t,
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bench", choices=("tree", "catalog", "budget", "child"))
    p.add_argument("--root", default="")
    p.add_argument("--files", type=int, default=20000)
    p.add_argument("--rows", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--skip-per-field", action="store_true")
    p.add_argument("--path", default="")
    p.add_argument("--dataset", default="")
    p.add_argument("--source", default="moss_shard")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--area", type=int, default=1024 * 1024)
    p.add_argument("--out", default="")
    args = p.parse_args()
    match args.bench:
        case "tree":
            res = bench_tree(args)
        case "catalog":
            res = bench_catalog(args)
        case "budget":
            res = bench_budget(args)
        case "child":
            res = child_open(args.path)
    print(json.dumps(res))
    if args.out:
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
