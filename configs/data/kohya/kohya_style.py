"""Data config for a kohya-style tagged dataset; drives scripts/data/inspect_data.py and
the other scripts/data/ passes (catalog_build, cache_text, cache_latents).

    kogine run scripts/data/inspect_data.py --config configs/data/kohya/kohya_style.py

Set data_root and WEIGHTS below (data_root is lowercase: a config-local name, not a
knob). Folders are `<repeats>_<name>` with `.txt` tag sidecars; the artist comes from
the folder name and is kept as the first caption token. The caches land in
`<data_root>/cache`, where a train config with the same DATASETS
path looks; a train run reads them when its key knobs (sizes, crops, flips, captions,
variants, seed, encoder specs) equal these. To cache for one train config, a data
config that `use_config`s it and adds the two encoder lines shares every key knob. See
docs/guides/datasets.md.
"""

data_root = "path/to/kohya_dataset"
WEIGHTS = "ComfyUI/models"

DATASETS = [
    {
        "source": "kohya",
        "path": data_root,
        "repeat_prefix": "first",
        "artist_from_folder": "name",
        "txt_as": "tags",
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
TEXT_ENCODER_KWARGS = {"path": WEIGHTS, "device": "cuda"}
IMAGE_ENCODER_KWARGS = {"path": WEIGHTS, "device": "cuda"}
