"""LoRA on an Intel Arc Pro B50 (16 GB, XPU): bf16 only, int8 base, the `low` profile.
Drives scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_xpu_b50.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")
use_config("../vram/vram_16g.py")
use_config("../precision/precision_bf16.py")

RUN_NAME = "lora-xpu-b50"
BASE_PRECISION = "int8"
MAX_TOKENS = 6144
TEXT_ENCODE = {"memory": "stream", "ahead": 4, "batch": 16, "max_tokens": 8192}
