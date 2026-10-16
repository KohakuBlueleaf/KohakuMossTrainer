"""Kohya-style tagged style dataset; drives scripts/data/*.py and scripts/train/moss.py.

    kogine run scripts/data/inspect_data.py --config configs/data/data_kohya_style.py

Set DATA_ROOT below (DATASETS and CACHE_DIR derive from it). Folders are
`<repeats>_<name>` with `.txt` tag sidecars; the artist comes from the folder name
and is kept as the first caption token.
"""

import os

DATA_ROOT = "path/to/kohya_dataset"
CACHE_DIR = os.path.join(DATA_ROOT, ".kohakumoss_cache")

DATASETS = [
    {
        "source": "kohya",
        "path": DATA_ROOT,
        "repeat_prefix": "first",
        "artist_from_folder": "name",
        "txt_as": "tags",
        "catalog": os.path.join(CACHE_DIR, "catalog.db"),
    }
]
RESOLUTION_MODE = "resize"
TARGET_AREA = 1024 * 1024
UPSCALE = "never"
ASPECT_RANGE = (0.33, 3.0)
BATCHING = "tokens"
MAX_TOKENS = 16384
OVERLONG = "drop"
CAPTION = {
    "template": "tags",
    "keep": ["artist"],
    "shuffle": True,
    "tag_dropout": 0.1,
    "underscores": True,
    "unescape": True,
}
CAPTION_VARIANTS = 4
TEXT_DROP = 0.1
LATENT_CACHE = "offline"
TEXT_CACHE = "offline"
FLIP = False
