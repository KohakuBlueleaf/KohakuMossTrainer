"""Data layer for scripts/train/moss.py: any image tree with natural-language `.txt` (or
`.caption` / `.json` `caption`) sidecars. `kohakumoss-config new` replaces the
placeholder path; see docs/guides/configs.md."""

DATASETS = [{"source": "folder", "path": "path/to/dataset", "txt_as": "nl"}]
ASPECT_RANGE = (0.33, 3.0)
CAPTION = {"template": "nl"}
