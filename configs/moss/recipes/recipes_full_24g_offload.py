"""Full finetune on one 24 GB GPU: trainable weight streaming with host AdamW masters,
activation offload. Drives scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_full_24g_offload.py
"""

from kohakuengine import use_config

use_config("../_base/base_full.py")

RUN_NAME = "full-24g-offload"
OPTIMIZER = "adamw"
EMA = 0.0
MEMORY_PROFILE = "minimal"
MEMORY_OVERRIDES = {"stream": "swap", "stream_resident": 4, "act_offload": True}
MAX_TOKENS = 8192
