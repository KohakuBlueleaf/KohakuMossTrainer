"""Parallel layer for scripts/train/moss.py: DDP over two GPUs (Linux). Composed after
the VRAM layer, whose memory profile runs unchanged (every profile runs under DDP). See
docs/guides/configs.md and docs/guides/multi-gpu.md."""

PARALLEL = "ddp"
DEVICES = (0, 1)
