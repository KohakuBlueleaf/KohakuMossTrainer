"""Data layer for scripts/train/moss.py: a dataset imported into the native moss_shard
format (catalog.db + shards/, see docs/concepts/data.md). `kohakumoss-config new`
replaces the placeholder path; see docs/guides/configs.md."""

DATASETS = [{"source": "moss_shard", "path": "path/to/dataset"}]
ASPECT_RANGE = (0.33, 3.0)
CAPTION = {
    "template": "tags",
    "keep": ["artist"],
    "shuffle": True,
    "tag_dropout": 0.1,
    "underscores": True,
    "unescape": True,
}
