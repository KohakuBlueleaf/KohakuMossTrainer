"""LoRA r32 with DDP over four RTX 5090 32 GB (Linux, nccl) at maximum speed: every rank
runs the one-GPU 5090 recipe (bf16 over a 16-bit base, 4608-token batches, the first
fifth of the blocks checkpointed). Drives scripts/train/moss.py. Measured:
docs/performance/training-5090.md.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_ddp_4x5090.py
"""

from kohakuengine import use_config

use_config("recipes_lora_5090.py")
use_config("../parallel/parallel_ddp.py")

RUN_NAME = "lora-ddp-4x5090"
DEVICES = (0, 1, 2, 3)
