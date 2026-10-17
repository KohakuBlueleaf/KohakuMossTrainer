"""LoRA with DDP over two GPUs (Linux, nccl; adapter-only reduce).
Mixed fp16-only + bf16 GPUs need PRECISION='fp16'. Drives scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_ddp.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")

RUN_NAME = "lora-ddp"
PARALLEL = "ddp"
DEVICES = (0, 1)
