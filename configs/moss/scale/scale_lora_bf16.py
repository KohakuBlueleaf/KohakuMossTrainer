"""Scaling-run layer for scripts/train/moss.py: LoRA (dim 32, attn-mlp) in bf16 on
static micro-batches (two ~1024^2 samples per 9216-token batch, packed first-fit),
offline latent / text caches, a resident block-checkpointed base (valid under every
PARALLEL kind), one metric row per step, the per-rank report and a profiled window.
The engine comes from --set (PARALLEL, DEVICES, GRAD_ACC, MICROBATCHES,
PIPELINE_SCHEDULE), WEIGHTS / DATASETS / CACHE_DIR / OUTPUT_DIR from a host layer that
uses this one. See docs/guides/multi-gpu.md (scaling runs).

    kogine run scripts/train/moss.py --config <host> --set PARALLEL=ddp --set DEVICES=0,1
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")
use_config("../data/data_moss_shard.py")
use_config("../precision/precision_bf16.py")

RUN_NAME = "scale"
TARGET_AREA = 1024 * 1024
MAX_TOKENS = 9216
BATCHING = "tokens_ffd"
MAX_SAMPLES = 4
TOKEN_BUDGET = "exact"
OVERLONG = "drop"
LATENT_CACHE = "offline"
TEXT_CACHE = "offline"
TEXT_DROP = 0.0
MEMORY_PROFILE = "ckpt-resident"
TUNE = "table"
STEPS = 30
SCHEDULE = {"mode": "constant", "warmup": 1, "end": -1}
LOG_EVERY = 1
METRICS_EVERY = 1
PROGRESS = False
SAVE_EVERY = 0
KEEP_LAST = 0
SAMPLE_EVERY = 0
VALIDATE_EVERY = 0
SAVE_FORMAT = "none"
TRACE_STEPS = (25, 28)
MICROBATCHES = 8
PIPELINE_SCHEDULE = "1f1b"
PIPELINE_SPLIT = "measured"
