"""VRAM layer for scripts/train/moss.py, full finetune on one 24 GB GPU: trainable weight
streaming (swap, four resident blocks) with host AdamW states, activation and gradient
offload; the text encoder streams its layers through a ring, 8 batches per pass.
Composed after configs/moss/algo/algo_full.py; see docs/guides/configs.md and
docs/concepts/profiles.md."""

OPTIMIZER = "adamw"
EMA = 0.0
MEMORY_PROFILE = "ckpt-ring4-actoff-optstream-gradoff"
MAX_TOKENS = 8192
TEXT_ENCODE = {"memory": "ring", "ahead": 8}
_BASE = "../_base/base_full.py"  # validation base of this layer
