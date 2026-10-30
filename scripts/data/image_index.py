"""Index KohakuClip image shards by key for a gbc dataset entry's `images` store.

    kogine run scripts/data/image_index.py --set ROOT=http://127.0.0.1:8317/shards \
        --set SHARD_LIST=shards.txt --set OUT=/data/index --set "KEEP=['packages/a.parquet']"

ROOT is the shard directory or an http(s) URL (e.g. `rclone serve http` over S3);
SHARD_LIST a text file of shard names (one per line), empty: every *.zip in ROOT (a
directory). KEEP: parquet files whose `key` column (rows of `KEEP_SOURCE` in their
`image_source` column, when set) limits the index. See
docs/concepts/data.md#kohakuclip-image-shards.
"""

import os
import time

import pyarrow.compute as pc
import pyarrow.parquet as pq

from kohakumoss_common.knobs import check_script
from kohakumoss_common.log import get_logger, setup
from kohakumoss_data.images import build_index

log = get_logger("scripts.image_index")

ROOT = ""
SHARD_LIST = ""
OUT = ""
KEEP = []
KEEP_SOURCE = ""
WORKERS = 32


def shard_names() -> list[str]:
    if SHARD_LIST:
        with open(SHARD_LIST, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return sorted(n for n in os.listdir(ROOT) if n.endswith(".zip"))


def kept_keys() -> set[int] | None:
    if not KEEP:
        return None
    keys: set[int] = set()
    for path in KEEP:
        table = pq.read_table(path, columns=["key", "image_source"])
        if KEEP_SOURCE:
            table = table.filter(pc.equal(table["image_source"], KEEP_SOURCE))
        keys.update(int(k) for k in table["key"].to_pylist())
    return keys


def main() -> None:
    setup()
    check_script(globals(), owners=(".",))
    if not ROOT or not OUT:
        raise SystemExit("set ROOT and OUT")
    names = shard_names()
    keep = kept_keys()
    log.info(f"{len(names)} shards, {'all' if keep is None else len(keep)} keys")
    began = time.perf_counter()
    rows = build_index(ROOT, names, OUT, keep, WORKERS)
    log.info(f"indexed {rows} images in {time.perf_counter() - began:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
