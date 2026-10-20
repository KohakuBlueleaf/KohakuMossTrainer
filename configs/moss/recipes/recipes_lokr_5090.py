"""LoKr factor 8 (w1 and w2 full) on one RTX 5090 32 GB at maximum speed: bf16 over a
16-bit base, LyCORIS's token-major LoKr kernels (lycoris-lora >= 4.1.3), one 1 MP
sample per 4608-token micro-batch, the first fifth of the blocks checkpointed, 64
micro-batches per optimizer step (294,912 tokens per step). Drives
scripts/train/moss.py. Measured: docs/performance/training-5090.md.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lokr_5090.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")
use_config("../algo/algo_lokr.py")
use_config("../precision/precision_bf16.py")

RUN_NAME = "lokr-f8-5090"
DIM = 10000
ALPHA = 1.0
ALGO_KWARGS = {"factor": 8, "full_matrix": True}
LR = 2e-3
BASE_PRECISION = "compute"
MAX_TOKENS = 4608
TOKEN_BUDGET = "render"
MEMORY_PROFILE = "manual"
CHECKPOINT = "partial:0.2"
TEXT_ENCODE = {"memory": "ring", "ahead": 8}
GRAD_ACC = 64
