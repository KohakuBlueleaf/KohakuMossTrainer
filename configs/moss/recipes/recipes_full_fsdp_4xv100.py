"""Full finetune on 4x V100 32 GB with FSDP2 (fp16 + scaler, AdamW16 states).
Drives scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_full_fsdp_4xv100.py
"""

from kohakuengine import use_config

use_config("../_base/base_full.py")

RUN_NAME = "full-fsdp-4xv100"
PRECISION = "fp16"
OPTIMIZER = "adamw"
PARALLEL = "fsdp"
DEVICES = (0, 1, 2, 3)
MEMORY_PROFILE = "balanced"
MAX_TOKENS = 8192
