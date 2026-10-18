"""Algorithm layer for scripts/train/moss.py: DoRA (LoRA r16 + weight decomposition,
detached norm). Composed after configs/moss/_base/base_lora.py; see
docs/guides/configs.md."""

ALGO = "lora"
DIM = 16
ALPHA = 16.0
DORA = True
DORA_DETACH = True
