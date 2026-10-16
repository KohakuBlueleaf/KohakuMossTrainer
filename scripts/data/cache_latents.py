"""Offline latent cache pass for the data config (every geometry and flip of epoch 0).

    kogine run scripts/data/cache_latents.py --config configs/data/data_kohya_style.py \
        --set IMAGE_ENCODER=<dotted factory returning an EncodeImages callable>

The factory gets IMAGE_ENCODER_KWARGS; its id must match LATENT_ENCODER_ID so the
trainer finds the entries. Missing keys only are encoded, so the pass resumes.
"""

import time

from kohakumoss_data.cache.build import build_latents
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.scripting import build_encoder, data_config

# ===================== DATA =====================
DATASETS = []
RESOLUTION_MODE = "resize"  # "resize" | "bucket" | "shifted_square" | "square_crop"
TARGET_AREA = 1024 * 1024
SIZE_STEP = 16
ASPECT_RANGE = (0.25, 4.0)
UPSCALE = "never"
FLIP = False
CACHE_DIR = ""
CACHE_DTYPE = "bf16"  # "bf16" | "fp8"
LATENT_ENCODER_ID = "flux2-vae"
SEED = 0

# ===================== ENCODER =====================
IMAGE_ENCODER = "kohakumoss_data.cache.random_encoders.RandomImageEncoder"
IMAGE_ENCODER_KWARGS = {}
ENCODE_BATCH = 8


def main() -> None:
    cfg = data_config(globals(), latent_cache="offline", text_cache="off")
    dataset = MossDataset(cfg)
    encode = build_encoder(IMAGE_ENCODER, IMAGE_ENCODER_KWARGS)
    start = time.perf_counter()

    def progress(n: int) -> None:
        rate = n / max(1e-9, time.perf_counter() - start)
        print(f"\r{n} latents ({rate:.1f}/s)", end="", flush=True)

    n = build_latents(dataset, encode, ENCODE_BATCH, progress)
    print(f"\nwrote {n} latents to {cfg.cache_dir}")


if __name__ == "__main__":
    main()
