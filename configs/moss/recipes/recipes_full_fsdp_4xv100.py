"""Full finetune on 4x V100 32 GB with FSDP2: fp16 + scaler with the online residual
fold (c = 1/16), fp32 AdamW masters. Drives scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_full_fsdp_4xv100.py
"""

from kohakuengine import use_config

use_config("../_base/base_full.py")
use_config("../precision/precision_fp16.py")
use_config("../parallel/parallel_fsdp.py")

RUN_NAME = "full-fsdp-4xv100"
DEVICES = (0, 1, 2, 3)
MAX_TOKENS = 8192
TEXT_ENCODE = {"memory": "ring", "dtype": "fp16", "ahead": 4}
