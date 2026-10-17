"""Full finetune on 4x V100 32 GB with FSDP2: fp16 + scaler with the online residual
fold (c = 1/16), fp32 AdamW masters. Drives scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_full_fsdp_4xv100.py
"""

from kohakuengine import use_config

use_config("../_base/base_full.py")

RUN_NAME = "full-fsdp-4xv100"
PRECISION = "fp16"
FP16_FIX = "online"
FP16_FIX_KWARGS = {"c": 1 / 16}
OPTIMIZER = "adamw"
PARALLEL = "fsdp"
DIST_BACKEND = "nccl"
DEVICES = (0, 1, 2, 3)
SAMPLE_EVERY = 0
MEMORY_PROFILE = "balanced"
MAX_TOKENS = 8192
