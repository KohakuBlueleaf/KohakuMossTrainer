"""Offline latent cache pass: every catalog row's centre-crop target (and its flip when
FLIP is on), for the data config.

    kogine run scripts/data/cache_latents.py --config configs/data/kohya/kohya_style.py \
        --set IMAGE_ENCODER=kohakumoss.encoders.vae_encoder

The factory gets IMAGE_ENCODER_KWARGS (for `kohakumoss.encoders.vae_encoder`:
{"path": <weights>/vae or a pipeline root, "device": "cuda"}; VAE_DTYPE fills "dtype",
VAE_ENCODE the placement "plan", "auto" when unset); its id
must match LATENT_ENCODER_ID so the trainer finds the entries, and its compute spec
(`encoder_spec`, the VAE dtype) enters the keys as LATENT_ENCODER_SPEC. Missing keys
only are encoded, so the pass resumes.
"""

import logging
import sys

from tqdm.auto import tqdm

from kohakumoss.encoders.specs import latent_encoder_kwargs
from kohakumoss_common.knobs import check_script
from kohakumoss_common.log import get_logger, setup
from kohakumoss_data.cache.build import build_latents
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.scripting import build_encoder, data_config

log = get_logger("scripts.cache_latents")

# ===================== DATA =====================
DATASETS = []
RESOLUTION_MODE = "resize"  # "resize" | "bucket" | "shifted_square" | "square_crop"
TARGET_AREA = 1024 * 1024
SIZE_STEP = 16
ASPECT_RANGE = (0.25, 4.0)
UPSCALE = "never"
CROP = "center"
MAX_TOKENS = 16384
FLIP = False
CACHE_DIR = ""
CACHE_DTYPE = "bf16"  # "bf16" | "fp16" | "fp32" | "fp8"
LATENT_ENCODER_ID = "flux2-vae"
BACKGROUND = "white"
RESAMPLE = "lanczos"
LATENT_DRAW = "mode"
SEED = 0

# ===================== ENCODER =====================
IMAGE_ENCODER = "kohakumoss.encoders.vae_encoder"
IMAGE_ENCODER_KWARGS = {"path": "", "device": "cuda"}  # path: <weights>/vae or a root
VAE_DTYPE = (
    "auto"  # "auto" | "bf16" | "fp16" | "fp32": the `dtype` of encoders taking one
)
VAE_ENCODE = (
    {}
)  # VaePlan fields as in training ("memory": "auto" when unset): the `plan` of encoders taking one
ENCODE_BATCH = 8


def main() -> None:
    setup()
    check_script(globals(), owners=("../train/moss.py", "."))
    encode = build_encoder(IMAGE_ENCODER, latent_encoder_kwargs(globals()))
    spec = getattr(encode, "encoder_spec", "")
    cfg = data_config(
        globals(), latent_cache="offline", text_cache="off", latent_encoder_spec=spec
    )
    dataset = MossDataset(cfg)
    quiet = not logging.getLogger("kohakumoss").isEnabledFor(logging.INFO)
    bar = tqdm(unit="latent", file=sys.stderr, disable=quiet)

    def progress(n: int) -> None:
        bar.update(n - bar.n)

    n = build_latents(dataset, encode, ENCODE_BATCH, progress)
    bar.close()
    log.info(f"wrote {n} latents to {cfg.cache_dir}")


if __name__ == "__main__":
    main()
