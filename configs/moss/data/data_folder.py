"""Data layer for scripts/train/moss.py: any image tree with `.txt` tag sidecars (or
`.json` with `tags` / `artist`). `kohakumoss-config new` replaces the placeholder path;
see docs/guides/configs.md."""

DATASETS = [{"source": "folder", "path": "path/to/dataset", "txt_as": "tags"}]
ASPECT_RANGE = (0.33, 3.0)
CAPTION = {
    "template": "tags",
    "keep": ["artist"],
    "shuffle": True,
    "tag_dropout": 0.1,
    "underscores": True,
    "unescape": True,
}
