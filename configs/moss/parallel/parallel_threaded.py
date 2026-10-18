"""Parallel layer for scripts/train/moss.py: single-process pipeline over two GPUs of
one host (mixed cards allowed; a 2080 Ti in the set needs PRECISION='fp16'), all
forwards then one backward over eight micro-batches per step. Composed after the VRAM
layer. See docs/guides/multi-gpu.md."""

PARALLEL = "threaded"
DEVICES = (0, 1)
PIPELINE_SCHEDULE = "gpipe"
MICROBATCHES = 8
SAMPLE_EVERY = 0
VALIDATE_EVERY = 0
MEMORY_PROFILE = "ckpt-resident"
MEMORY_OVERRIDES = {}
