"""Precision layer for scripts/sample/sample.py: fp16 transformer for fp16-only GPUs (V100, 2080 Ti),
with the online residual-stream fix (c = 1/16). See docs/guides/configs.md and
docs/concepts/numerics.md."""

PRECISION = "fp16"
FP16_FIX = "online"
FP16_FIX_KWARGS = {"c": 1 / 16}
