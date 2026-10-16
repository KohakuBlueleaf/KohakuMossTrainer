"""Scan every dataset in DATASETS once and write its catalog (or import it into shards).

    kogine run scripts/data/catalog_build.py --config configs/data/data_kohya_style.py

Each dataset entry may set `catalog` (path of the catalog.db to write / reuse later)
and `import_to` (directory: copy the images into the native moss_shard format).
"""

import json
import os

from kohakumoss_data.dataset import build_source
from kohakumoss_data.format.catalog import Catalog, write_catalog
from kohakumoss_data.importer import import_source
from kohakumoss_data.stats import catalog_stats

# ===================== DATA =====================
DATASETS = []  # same entries as the train script


def main() -> None:
    for spec in DATASETS:
        source = build_source(spec)
        if spec.get("import_to"):
            n = import_source(source, spec["import_to"])
            print(f"imported {n} images -> {spec['import_to']}")
            continue
        rows = list(source.scan())
        path = spec.get("catalog")
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            write_catalog(path, rows)
            print(f"wrote {len(rows)} rows -> {path}")
        print(json.dumps(catalog_stats(Catalog(rows)), indent=1))


if __name__ == "__main__":
    main()
