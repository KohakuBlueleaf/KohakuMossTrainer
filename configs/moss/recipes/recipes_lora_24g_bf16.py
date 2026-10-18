"""LoRA r32 on a 24 GB bf16 GPU (3090 / 4090 / 50xx): packed 1 MP batches, block
checkpointing. Drives scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_24g_bf16.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")
use_config("../vram/vram_24g.py")
use_config("../precision/precision_bf16.py")

RUN_NAME = "lora-24g-bf16"
