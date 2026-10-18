"""Precision layer for scripts/train/moss.py: fp16 compute for GPUs without bf16 (V100,
2080 Ti) with the online residual-stream fix (c = 1/16). See docs/guides/configs.md and
docs/concepts/numerics.md."""

PRECISION = "fp16"
FP16_FIX = "online"
FP16_FIX_KWARGS = {"c": 1 / 16}
