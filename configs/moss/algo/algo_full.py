"""Algorithm layer for scripts/train/moss.py: full finetune with the tech-report optimiser
(Muon + AdamW, wd 0.01, decay-to-one on the last block's q/k norm, clip 1.0) and EMA
weights, on the resident block-checkpointed `ckpt-resident` profile (per-parameter
steps after backward). Composed after configs/moss/_base/base_lora.py; see
docs/guides/configs.md and docs/concepts/profiles.md."""

ALGO = "full"
SAVE_FORMAT = "comfy"
OPTIMIZER = "muon"
LR = 5e-5
WD_RULES = "moss"
SCHEDULE = {"mode": "cosine", "warmup": 500, "end": -1, "min_value": 0.1}
EMA = 0.9999
MEMORY_PROFILE = "ckpt-resident"
