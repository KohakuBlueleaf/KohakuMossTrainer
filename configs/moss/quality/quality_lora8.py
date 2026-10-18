"""Per-card training quality test, LoRA r8. Drives scripts/train/moss.py:
400 steps, constant LR after 20 warmup steps, checkpoints every 100 steps, seed 0,
online text cache with the streamed text encoder, offline latent cache.

    kogine run scripts/train/moss.py --config configs/moss/quality/quality_lora8.py
    kogine run scripts/train/moss.py --config configs/moss/quality/quality_lora8.py \
        --set PRECISION=fp16 --set FP16_FIX=online --set FP16_FIX_KWARGS={"c":0.0625}

fp16 from the fp16-fix checkpoint (the fold comes from the file metadata; FP16_FIX stays
"none"; text encoder and VAE from the base folder):

    kogine run scripts/train/moss.py --config configs/moss/quality/quality_lora8.py \
        --set PRECISION=fp16 \
        --set WEIGHTS=<MOSS-Image>/fp16fix/moss_image_base_fp16fix_c0.0625.safetensors \
        --set TEXT_ENCODER=<MOSS-Image>/base/text_encoder --set VAE=<MOSS-Image>/base/vae
"""

from kohakuengine import use_config

use_config("../_base/base_lora.py")
use_config("../algo/algo_lora.py")
use_config("../vram/vram_auto.py")
use_config("../precision/precision_bf16.py")

RUN_NAME = "quality-lora8"
OUTPUT_DIR = "runs/quality"
DIM = 8
ALPHA = 8.0
LR = 5e-4
OPTIMIZER = "adamw"
STEPS = 400
SCHEDULE = {"mode": "constant", "warmup": 20, "end": -1}
SAVE_EVERY = 100
KEEP_LAST = 0
SEED = 0
SAMPLE_EVERY = 0
LATENT_CACHE = "offline"
TEXT_CACHE = "online"
TEXT_ENCODE = {"memory": "ring"}
