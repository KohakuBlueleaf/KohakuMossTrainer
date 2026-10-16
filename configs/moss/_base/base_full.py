"""Shared full-finetune defaults: the tech-report optimiser (Muon + AdamW, wd 0.01,
decay-to-one on the last block's q/k norm, clip 1.0) and EMA weights."""

from kohakuengine import use_config

use_config("base_lora.py")

ALGO = "full"
EXPORT_FORMATS = ("merged",)
OPTIMIZER = "muon"
LR = 5e-5
WD_RULES = "moss"
SCHEDULE = {"mode": "cosine", "warmup": 500, "end": -1, "min_value": 0.1}
EMA = 0.9999
