"""Full finetune with DDP over four RTX 5090 32 GB (Linux, nccl) at maximum speed: every
rank runs the one-GPU 5090 full-finetune recipe (weight swap with eight resident blocks,
activation offload, streamed AdamW states), the ranks sharing one host copy of the
masters. Drives scripts/train/moss.py. Measured: docs/performance/training-5090.md.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_full_ddp_4x5090.py
"""

from kohakuengine import use_config

use_config("recipes_full_5090.py")
use_config("../parallel/parallel_ddp.py")

RUN_NAME = "full-ddp-4x5090"
DEVICES = (0, 1, 2, 3)
