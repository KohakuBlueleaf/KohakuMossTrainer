"""Data layer for scripts/train/moss.py: a JSON-lines manifest, one record per line with
the image path under `image` / `file` / `path` / `file_name` (relative to `root`, default
the manifest's folder), captions under `caption` / `text` / `caption_nl` or `tags`, and
optional `artist`, `rating`, `weight`, `group`. Natural-language captions, tags where a
record has none. `kohakumoss-config new` replaces the placeholder path (the .jsonl file);
see docs/guides/configs.md and docs/concepts/data.md."""

DATASETS = [{"source": "jsonl", "path": "path/to/dataset/train.jsonl", "root": None}]
ASPECT_RANGE = (0.33, 3.0)
CAPTION = {"template": "nl"}
