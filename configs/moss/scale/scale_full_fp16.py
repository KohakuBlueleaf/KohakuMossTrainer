"""Scaling-run layer for scripts/train/moss.py: full finetune in fp16 (online residual
fix) with AdamW (fp32 masters, no EMA) on the scaling data knobs; meant for
PARALLEL='fsdp' over four 32 GB V100s. See docs/guides/multi-gpu.md (scaling runs)."""

from kohakuengine import use_config

use_config("scale_lora_fp16.py")
use_config("../algo/algo_full.py")

OPTIMIZER = "adamw"
LR = 1e-5
EMA = 0.0
SAVE_FORMAT = "none"
MEMORY_PROFILE = "ckpt-resident"
SCHEDULE = {"mode": "constant", "warmup": 1, "end": -1}
