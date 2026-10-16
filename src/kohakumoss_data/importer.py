"""Import any source into the native `moss_shard` format (shards + catalog.db)."""

import os

from kohakumoss_data.format.catalog import write_catalog
from kohakumoss_data.format.shard import SHARD_BYTES, ShardWriter, codec_of
from kohakumoss_data.seeds import content_hash


def import_source(source, out_dir: str, shard_bytes: int = SHARD_BYTES) -> int:
    """Copy every image of `source` into `out_dir`; rows keep their metadata, gain
    `shard:index` locators and a content hash. Returns the image count."""
    os.makedirs(out_dir, exist_ok=True)
    writer = ShardWriter(out_dir, shard_bytes)
    rows = []
    try:
        for row in source.scan():
            data = source.open(row.locator)
            digest = content_hash(data)
            row.extra = {**row.extra, "hash": digest, "origin": row.locator}
            row.locator = writer.add(
                data,
                min(row.w, 65535),
                min(row.h, 65535),
                codec_of(row.extra["origin"]),
                row.has_alpha,
                int(digest[:16], 16),
            )
            rows.append(row)
    finally:
        writer.close()
    return write_catalog(os.path.join(out_dir, "catalog.db"), rows)
