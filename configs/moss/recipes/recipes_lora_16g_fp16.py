"""LoRA on fp16-only GPUs (V100 16/32 GB, 2080 Ti 22 GB): fp16 compute with the
dynamic loss scaler, fp32 LoRA masters, TileLang kernels, the online residual-stream
fix (c = 1/16, docs/concepts/numerics.md) and the 16 GB VRAM layer.
Drives scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_16g_fp16.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")
use_config("../vram/vram_16g.py")
use_config("../precision/precision_fp16.py")

RUN_NAME = "lora-16g-fp16"
