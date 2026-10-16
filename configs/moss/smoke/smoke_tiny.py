"""Smoke run on a random tiny MOSS with synthetic data (no downloads); drives
scripts/train/moss.py.

    kogine run scripts/train/moss.py --config configs/moss/smoke/smoke_tiny.py
"""

WEIGHTS = "random"
VARIANT = "tiny"
DATASETS = [{"source": "synthetic", "count": 64}]
RESOLUTION_MODE = "bucket"
TARGET_AREA = 256 * 256
BATCH_SIZE = 4
LATENT_CACHE = "off"
TEXT_CACHE = "off"
NUM_WORKERS = 0
DIM = 4
ALPHA = 4.0
STEPS = 20
SCHEDULE = {"mode": "cosine", "warmup": 2, "end": -1}
LOG_EVERY = 1
SAVE_EVERY = 10
PRECISION = "fp32"
OUTPUT_DIR = "runs/smoke"
RUN_NAME = "tiny"
