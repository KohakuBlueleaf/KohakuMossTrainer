"""Smoke run on a random tiny MOSS with synthetic data; drives scripts/train/moss.py.

kogine run scripts/train/moss.py --config configs/moss/smoke/smoke_tiny.py
"""

WEIGHTS = "random"
VARIANT = "tiny"
DATASETS = [{"source": "synthetic", "count": 64}]
LATENT_CACHE = "off"
TEXT_CACHE = "off"
MAX_TOKENS = 1024
STEPS = 20
LOG_EVERY = 1
SAVE_EVERY = 10
PRECISION = "fp32"
