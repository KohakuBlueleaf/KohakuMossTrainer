"""The native format as a source: `catalog.db` + `shards/`."""

import os
from collections.abc import Iterator

from kohakumoss_data.format.catalog import Catalog, read_catalog
from kohakumoss_data.format.shard import ShardReader
from kohakumoss_data.registry import SOURCES
from kohakumoss_data.types import CatalogRow


@SOURCES.register("moss_shard")
class MossShardSource:
    """A dataset directory written by `kohakumoss_data.importer.import_source`."""

    def __init__(self, path: str):
        self.path = path
        self.reader = ShardReader(path)

    def catalog(self) -> Catalog:
        """The dataset's own catalog; FileNotFoundError naming what is missing,
        ValueError naming the import command for a catalog.db of another layout."""
        if not os.path.isdir(self.reader.dir):
            raise FileNotFoundError(
                f"moss_shard {self.path}: no shards/ folder (re-run the import that "
                "wrote this dataset)"
            )
        return read_catalog(os.path.join(self.path, "catalog.db"))

    def scan(self) -> Iterator[CatalogRow]:
        yield from self.catalog().iter_rows()

    def open(self, locator: str) -> bytes:
        return self.reader.read(locator)
