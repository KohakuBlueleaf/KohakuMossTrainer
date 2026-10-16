"""DoRA (LoRA r16 + weight decomposition, detached norm). Drives scripts/train/moss.py.

kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_dora.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")

RUN_NAME = "dora-r16"
ALGO = "lora"
DIM = 16
ALPHA = 16.0
DORA = True
DORA_DETACH = True
