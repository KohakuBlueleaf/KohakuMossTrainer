"""Scan every dataset in DATASETS once and write its catalog (or import it into shards).

    kogine run scripts/data/catalog_build.py --config configs/data/kohya/kohya_style.py

Each dataset entry may set `catalog` (path of the catalog.db to write / reuse later)
and `import_to` (directory: copy the images into the native moss_shard format).
"""

import json
import os

from kohakumoss_common.knobs import check_script
from kohakumoss_common.log import get_logger, setup
from kohakumoss_data.dataset import build_source
from kohakumoss_data.importer import import_source
from kohakumoss_data.scan.runner import scan_catalog
from kohakumoss_data.stats import catalog_stats

log = get_logger("scripts.catalog_build")

# ===================== DATA =====================
DATASETS = []  # same entries as the train script


def main() -> None:
    setup()
    check_script(globals(), owners=("../train/moss.py", "."))
    for entry in DATASETS:
        source = build_source(entry)
        if entry.get("import_to"):
            n = import_source(source, entry["import_to"])
            log.info(f"imported {n} images -> {entry['import_to']}")
            continue
        path = entry.get("catalog")
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            catalog = scan_catalog(source, path, kind="catalog")
            log.info(f"wrote {len(catalog)} rows -> {path}")
        else:
            catalog = scan_catalog(source, None)
        log.info(json.dumps(catalog_stats(catalog), indent=1))


if __name__ == "__main__":
    main()
