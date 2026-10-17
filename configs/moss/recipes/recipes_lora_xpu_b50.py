"""LoRA on an Intel Arc Pro B50 (16 GB, XPU): bf16 only, int8 base, the `low` profile.
Drives scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_xpu_b50.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")

RUN_NAME = "lora-xpu-b50"
PRECISION = "bf16"
MEMORY_PROFILE = "low"
MEMORY_OVERRIDES = {"base_precision": "int8"}
MAX_TOKENS = 6144
