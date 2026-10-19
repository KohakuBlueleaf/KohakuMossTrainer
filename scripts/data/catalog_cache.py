"""Show, drop or compact the catalog cache files of the datasets in DATASETS.

    kogine run scripts/data/catalog_cache.py --config configs/data/kohya/kohya_style.py

ACTION: "show" (row counts and generation per dataset), "drop" (delete the files: the
next run scans cold), "compact" (rewrite live rows only, in walk order, new ids),
"drop_all" (every catalog file under the cache dir). Run with no training job using
the cache (each file has one writer).
"""

import json
import os

from kohakumoss_common.knobs import check_script
from kohakumoss_common.log import get_logger, setup
from kohakumoss_data.catalog.vault import CatalogVault
from kohakumoss_data.config import catalog_cache_dir
from kohakumoss_data.dataset import build_source, source_kwargs
from kohakumoss_data.scan.lock import read_ready
from kohakumoss_data.scan.manage import compact, drop, drop_all
from kohakumoss_data.scan.paths import scan_path
from kohakumoss_data.scripting import data_config

log = get_logger("scripts.catalog_cache")

# ===================== DATA =====================
DATASETS = []  # same entries as the train script
CACHE_DIR = ""  # "" = <dataset>/cache
ACTION = "show"  # "show" | "drop" | "compact" | "drop_all"
DATA_SCAN_WORKERS = 8


def _show(path: str) -> dict:
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    vault = CatalogVault(path)
    live = int(vault.read_fixed()["live"].sum())
    return {"path": path, **vault.info(), "live": live, "ready": read_ready(path)}


def main() -> None:
    setup()
    check_script(globals(), owners=("../train/moss.py", "."))
    cfg = data_config(globals())
    cache_dir = catalog_cache_dir(cfg)
    if not cache_dir:
        raise SystemExit("no cache dir: set CACHE_DIR or a dataset `path`")
    if ACTION == "drop_all":
        for path in drop_all(cache_dir):
            log.info(f"dropped {path}")
        return
    for entry in DATASETS:
        path = scan_path(cache_dir, source_kwargs(entry))
        match ACTION:
            case "show":
                log.info(json.dumps(_show(path)))
            case "drop":
                log.info(f"{'dropped' if drop(path) else 'no file'} {path}")
            case "compact":
                n = compact(build_source(entry), path, DATA_SCAN_WORKERS)
                log.info(f"compacted {path}: {n} rows")
            case _:
                raise SystemExit(f"ACTION={ACTION!r} is not show|drop|compact|drop_all")


if __name__ == "__main__":
    main()
