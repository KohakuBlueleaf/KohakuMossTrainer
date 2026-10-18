"""Data layer for scripts/train/moss.py: a parquet table, one row per image: an image path
column (`image_col`, relative to `root`, default the file's folder) or an encoded-image
bytes column (`bytes_col`), captions in `caption` / `text` / `caption_nl` or `tags`,
optional `artist`, `rating`, `weight`, `group` columns. Natural-language captions, tags
where a row has none; needs the `parquet` extra. `kohakumoss-config new` replaces the
placeholder path (the .parquet file); see docs/guides/configs.md and
docs/concepts/data.md."""

DATASETS = [
    {
        "source": "parquet",
        "path": "path/to/dataset/train.parquet",
        "image_col": "image",
        "bytes_col": None,
        "root": None,
    }
]
ASPECT_RANGE = (0.33, 3.0)
CAPTION = {"template": "nl"}
