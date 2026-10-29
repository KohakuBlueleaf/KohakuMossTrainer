"""LoKr factor 8 (w1 and w2 full) with DDP over four RTX 5090 32 GB (Linux, nccl) at
maximum speed: the one-GPU LoKr 5090 recipe per rank, 16 micro-batches per rank per
optimizer step (64 x 4608 = 294,912 tokens per step). Drives scripts/train/moss.py.
Measured: docs/performance/training-5090.md.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lokr_ddp_4x5090.py
"""

from kohakuengine import use_config

use_config("recipes_lokr_5090.py")
use_config("../parallel/parallel_ddp.py")

RUN_NAME = "lokr-f8-ddp-4x5090"
DEVICES = (0, 1, 2, 3)
GRAD_ACC = 16
