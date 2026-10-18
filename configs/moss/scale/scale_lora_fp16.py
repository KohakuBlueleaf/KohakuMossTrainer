"""Scaling-run layer for scripts/train/moss.py on fp16-only GPUs (V100, 2080 Ti): the
bf16 scaling layer with fp16 compute and the online residual-stream fix (c = 1/16).
See docs/guides/multi-gpu.md (scaling runs)."""

from kohakuengine import use_config

use_config("scale_lora_bf16.py")
use_config("../precision/precision_fp16.py")
