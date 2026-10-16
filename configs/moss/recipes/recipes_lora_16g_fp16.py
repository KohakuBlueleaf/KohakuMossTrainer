"""LoRA on fp16-only GPUs (V100 16/32 GB, 2080 Ti 22 GB): fp16 compute with the
dynamic loss scaler, fp32 LoRA masters, TileLang kernels, and the online residual-stream
fix (c = 1/16 keeps the base's peak stream activation of ~62k under the fp16 max).
Drives scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/recipes/recipes_lora_16g_fp16.py
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")

RUN_NAME = "lora-16g-fp16"
PRECISION = "fp16"
FP16_FIX = "online"
FP16_FIX_KWARGS = {"c": 1 / 16}
MEMORY_PROFILE = "low"
MAX_TOKENS = 8192
