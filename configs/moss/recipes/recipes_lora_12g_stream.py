"""LoRA on 12 GB cards: int8 frozen base, H2D-only weight streaming, activation offload.
Drives scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_12g_stream.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")
use_config("../vram/vram_12g.py")

RUN_NAME = "lora-12g-stream"
BASE_PRECISION = "int8"
