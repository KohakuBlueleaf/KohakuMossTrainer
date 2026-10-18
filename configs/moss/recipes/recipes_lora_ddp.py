"""LoRA with DDP over two 24 GB bf16 GPUs (Linux, nccl; adapter-only reduce), with the
24 GB VRAM and bf16 precision layers. Mixed fp16-only + bf16 GPUs need PRECISION='fp16';
other cards: `kohakumoss-config new train --gpus 2`. Drives scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_ddp.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")
use_config("../vram/vram_24g.py")
use_config("../precision/precision_bf16.py")
use_config("../parallel/parallel_ddp.py")

RUN_NAME = "lora-ddp"
