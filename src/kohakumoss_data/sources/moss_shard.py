"""The native format as a source: `catalog.db` + `shards/`."""

import os
from collections.abc import Iterator

from kohakumoss_data.format.catalog import read_catalog
from kohakumoss_data.format.shard import ShardReader
from kohakumoss_data.registry import SOURCES
from kohakumoss_data.types import CatalogRow


@SOURCES.register("moss_shard")
class MossShardSource:
    """A dataset directory written by `kohakumoss_data.importer.import_source`."""

    def __init__(self, path: str):
        self.path = path
        self.reader = ShardReader(path)

    def scan(self) -> Iterator[CatalogRow]:
        yield from read_catalog(os.path.join(self.path, "catalog.db")).rows

    def open(self, locator: str) -> bytes:
        return self.reader.read(locator)
