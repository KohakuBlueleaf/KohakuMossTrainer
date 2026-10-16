"""Offline text-embedding cache pass: every caption variant of every row + empty prompt.

    kogine run scripts/data/cache_text.py --config configs/data/data_kohya_style.py \
        --set TEXT_ENCODER=<dotted factory returning an EncodeText callable>

The factory gets TEXT_ENCODER_KWARGS; its id must match TEXT_ENCODER_ID. Captions are
rendered with the same CAPTION template, CAPTION_VARIANTS and SEED as training.
"""

import time

from kohakumoss_data.cache.build import build_text
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.scripting import build_encoder, data_config

# ===================== DATA =====================
DATASETS = []
CAPTION = {"template": "tags", "keep": ["artist"], "shuffle": True, "tag_dropout": 0.0}
CAPTION_VARIANTS = 4
CACHE_DIR = ""
CACHE_DTYPE = "bf16"  # "bf16" | "fp8"
TEXT_ENCODER_ID = "qwen3.5-9b"
SEED = 0

# ===================== ENCODER =====================
TEXT_ENCODER = "kohakumoss_data.cache.random_encoders.RandomTextEncoder"
TEXT_ENCODER_KWARGS = {}
ENCODE_BATCH = 16


def main() -> None:
    cfg = data_config(globals(), latent_cache="off", text_cache="offline")
    dataset = MossDataset(cfg)
    encode = build_encoder(TEXT_ENCODER, TEXT_ENCODER_KWARGS)
    start = time.perf_counter()

    def progress(n: int) -> None:
        rate = n / max(1e-9, time.perf_counter() - start)
        print(f"\r{n} captions ({rate:.1f}/s)", end="", flush=True)

    n = build_text(dataset, encode, ENCODE_BATCH, progress)
    print(f"\nwrote {n} embeddings to {cfg.cache_dir}")


if __name__ == "__main__":
    main()
