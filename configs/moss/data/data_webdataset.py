"""Data layer for scripts/train/moss.py: WebDataset shards (uncompressed .tar), members
`<key>.<image ext>` with `<key>.txt` (tags, or natural language with `txt_as="nl"`)
and / or `<key>.json` (`tags`, `caption`, `artist`, `rating`). `kohakumoss-config new`
replaces the placeholder path (a glob of the .tar files); see docs/guides/configs.md and
docs/concepts/data.md."""

DATASETS = [{"source": "webdataset", "path": "path/to/shards/*.tar", "txt_as": "tags"}]
ASPECT_RANGE = (0.33, 3.0)
CAPTION = {
    "template": "tags",
    "keep": ["artist"],
    "shuffle": True,
    "tag_dropout": 0.1,
    "underscores": True,
    "unescape": True,
}
