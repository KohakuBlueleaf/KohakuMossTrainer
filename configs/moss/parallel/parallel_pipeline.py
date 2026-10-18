"""Parallel layer for scripts/train/moss.py: process-group pipeline over two GPUs
(Linux), 1F1B with eight micro-batches per step, the stage split measured at startup;
static micro-batches (first-fit token batches with a sample cap, exact text lengths
from the offline text cache). Composed after the VRAM layer. See
docs/guides/multi-gpu.md."""

PARALLEL = "pipeline"
DEVICES = (0, 1)
PIPELINE_SCHEDULE = "1f1b"
PIPELINE_SPLIT = "measured"
MICROBATCHES = 8
BATCHING = "tokens_ffd"
MAX_SAMPLES = 4
TOKEN_BUDGET = "exact"
TEXT_CACHE = "offline"
SAMPLE_EVERY = 0
VALIDATE_EVERY = 0
MEMORY_PROFILE = "ckpt-resident"
MEMORY_OVERRIDES = {}
