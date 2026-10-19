"""LoRA r32 on one RTX 5090 32 GB at maximum speed: bf16 over a 16-bit base, one 1 MP
sample per 4608-token batch (caption lengths from the offline text cache), the first fifth
of the blocks checkpointed. Drives scripts/train/moss.py. Measured:
docs/performance/training-5090.md.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_5090.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")
use_config("../precision/precision_bf16.py")

RUN_NAME = "lora-5090"
BASE_PRECISION = "compute"
MAX_TOKENS = 4608
TOKEN_BUDGET = "render"
MEMORY_PROFILE = "manual"
CHECKPOINT = "partial:0.2"
TEXT_ENCODE = {"memory": "ring", "ahead": 8}
