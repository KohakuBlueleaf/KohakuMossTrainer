"""Data layer for scripts/train/moss.py: a kohya-style tree (`<repeats>_<name>/` folders,
`.txt` tag sidecars; `include_root: True` also reads images in the dataset root, repeat
1), the artist from the folder name kept as the first caption token.
`kohakumoss-config new` replaces the placeholder path; see docs/guides/configs.md."""

DATASETS = [
    {
        "source": "kohya",
        "path": "path/to/dataset",
        "repeat_prefix": "first",
        "artist_from_folder": "name",
        "txt_as": "tags",
        "include_root": False,
    }
]
ASPECT_RANGE = (0.33, 3.0)
CAPTION = {
    "template": "tags",
    "keep": ["artist"],
    "shuffle": True,
    "tag_dropout": 0.1,
    "underscores": True,
    "unescape": True,
}
