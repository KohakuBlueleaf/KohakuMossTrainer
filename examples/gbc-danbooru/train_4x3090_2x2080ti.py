"""LoKr factor 8 on a danbooru GBC dataset: four RTX 3090 train (DDP, 4 x 4 = 16 samples
per step), two RTX 2080 Ti run the text encoder and the VAE in the encode feed. The DiT
and the text encoder are the int8 ConvRot ComfyUI files with int8 compute; the rest runs
in fp16 with the online fp16 fix, so the LoKr applies to any MOSS-Image weights.
7,000 steps of 16 = 112k samples, about one epoch of 113k.

Set WEIGHTS and DATASETS[0]["path"] (a config of your own can use_config this file and
override them), then, with the 3090s at CUDA 0-3 and the 2080 Tis at 4-5:

    kogine run scripts/train/moss.py --config examples/gbc-danbooru/train_4x3090_2x2080ti.py
"""

from kohakuengine import use_config

use_config("../../configs/moss/_base/base_lora.py")
use_config("../../configs/moss/algo/algo_lokr.py")
use_config("../../configs/moss/precision/precision_fp16.py")
use_config("../../configs/moss/parallel/parallel_ddp.py")
use_config("caption_danbooru.py")

RUN_NAME = "lokr-f8-gbc-danbooru"
WEIGHTS = "/path/to/MOSS-Image/comfy"  # a ComfyUI folder with the int8 ConvRot files
DATASETS = [
    {"source": "gbc", "path": "/path/to/danbooru-gbc-27b"}
]  # CAPTION: the layer

# data: native aspect, 6144 tokens per micro-batch (2048 text + up to 4096 image)
RESOLUTION_MODE = "resize"
FLIP = False
MAX_TOKENS = 6144
MAX_TEXT_TOKENS = 2048
OVERLONG = "truncate_text"
LATENT_CACHE = "off"
TEXT_CACHE = "off"
TEXT_DROP = 0.1

# LoKr factor 8, w1 and w2 full
ALGO_KWARGS = {"factor": 8, "full_matrix": True}
LR = 1e-4
SCHEDULE = {"mode": "cosine", "warmup": 100, "end": -1, "min_value": 0.01}

# int8 ConvRot DiT and text encoder, int8 GEMMs; fp16 VAE
BASE_PRECISION = "auto"
INT8_COMPUTE = True
INT8_DGRAD = "int8"
TEXT_ENCODE = {"memory": "resident", "int8_compute": True}
VAE_DTYPE = "fp16"
VAE_ENCODE = {"memory": "resident"}
MEMORY_PROFILE = "manual"
CHECKPOINT = "partial:0.75"

# 4x RTX 3090 train, 2x RTX 2080 Ti encode
DEVICES = (0, 1, 2, 3)
ENCODE_DEVICES = (4, 5)
ENCODE_DEPTH = 32
ENCODE_CHUNK = 8
GRAD_ACC = 4

STEPS = 7000
SAVE_EVERY = 500
SAMPLE_EVERY = 0
VALIDATE_EVERY = 0
