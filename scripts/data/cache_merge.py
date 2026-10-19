"""Fold the online caches' per-rank shards (`latents.r<rank>.kv`, `text.r<rank>.kv`)
into the main `latents.db` / `text.db` and delete them.

    kogine run scripts/data/cache_merge.py --config configs/data/kohya/kohya_style.py

Run with no training job writing the cache (the shards' writers must have exited).
"""

import os

from kohakumoss_common.knobs import check_script
from kohakumoss_common.log import get_logger, setup
from kohakumoss_data.cache.store import merge_shards, shard_paths
from kohakumoss_data.config import resolved_cache_dir
from kohakumoss_data.scripting import data_config

log = get_logger("scripts.cache_merge")

# ===================== DATA =====================
DATASETS = []
CACHE_DIR = ""  # "" = <dataset>/cache
LATENT_CACHE = "online"
TEXT_CACHE = "online"


def main() -> None:
    setup()
    check_script(globals(), owners=("../train/moss.py", "."))
    cache_dir = resolved_cache_dir(data_config(globals()))
    if not cache_dir:
        raise SystemExit("no cache dir: set CACHE_DIR or a dataset `path`")
    for name in ("latents.db", "text.db"):
        path = os.path.join(cache_dir, name)
        shards = shard_paths(path)
        if not shards:
            continue
        log.info(
            f"{name}: merged {merge_shards(path)} entries from {len(shards)} shards"
        )


if __name__ == "__main__":
    main()
