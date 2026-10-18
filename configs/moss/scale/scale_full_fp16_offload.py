"""Scaling-run layer for scripts/train/moss.py: the single-GPU reference of the fp16
full-finetune scaling runs (the full model does not fit one card resident): trainable
weight streaming with host AdamW states, activation and gradient offload. See
docs/guides/multi-gpu.md (scaling runs)."""

from kohakuengine import use_config

use_config("scale_full_fp16.py")

MEMORY_PROFILE = "ckpt-stream4-actoff-optstream-gradoff"
