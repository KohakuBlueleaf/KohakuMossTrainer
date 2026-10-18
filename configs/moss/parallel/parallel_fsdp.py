"""Parallel layer for scripts/train/moss.py: full finetune with FSDP2 (Linux, nccl, fp32
AdamW masters) on resident, block-checkpointed weights; replaces the VRAM layer's
profile. See docs/guides/configs.md and docs/guides/multi-gpu.md."""

PARALLEL = "fsdp"
DIST_BACKEND = "nccl"
DEVICES = (0, 1)
OPTIMIZER = "adamw"
SAMPLE_EVERY = 0
MEMORY_PROFILE = "ckpt-resident"
MEMORY_OVERRIDES = {}
