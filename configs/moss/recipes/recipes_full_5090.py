"""Full finetune on one RTX 5090 32 GB at maximum speed: block-checkpointed weight swap
with eight resident blocks, activation offload and streamed host AdamW states, 36864
tokens per step. Drives scripts/train/moss.py. Measured:
docs/performance/training-5090.md.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_full_5090.py
"""

from kohakuengine import use_config

use_config("../_base/base_full.py")
use_config("../precision/precision_bf16.py")

RUN_NAME = "full-5090"
OPTIMIZER = "adamw"
EMA = 0.0
MEMORY_PROFILE = "ckpt-ring8-actoff-optstream"
MAX_TOKENS = 36864
TEXT_ENCODE = {"memory": "ring", "ahead": 8}
