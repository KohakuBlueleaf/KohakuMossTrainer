"""Loader throughput: batches/s, images/s, joint tokens/s across worker counts.

    kogine run scripts/bench/data/loader.py --config configs/data/kohya/kohya_style.py

DATASETS comes from the data config. Results go to `OUT_DIR/loader_<mode>.json`, one
record per worker count, written after each.
"""

import json
import os
import time

from kohakumoss_common.knobs import check_script
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.loader.stateless import Loader
from kohakumoss_data.scripting import data_config

# ===================== DATA =====================
DATASETS = []  # [{"source": ..., "path": ...}] (the data config's)
RESOLUTION_MODE = "resize"
TARGET_AREA = 1024 * 1024
BATCHING = "tokens"
MAX_TOKENS = 16384
OVERLONG = "drop"
BATCH_SIZE = 4
LATENT_CACHE = "off"
TEXT_CACHE = "off"
CACHE_DIR = ""
SEED = 0

# ===================== BENCH =====================
WORKERS = (0, 2, 4, 8)
WARMUP = 4
BATCHES = 32
OUT_DIR = "runs/bench/data"


def run(dataset, workers: int) -> dict:
    loader = Loader(dataset, num_workers=workers, pin_memory=True)
    it = iter(loader)
    for _ in range(WARMUP):
        next(it)
    images = tokens = 0
    start = time.perf_counter()
    for _ in range(BATCHES):
        batch = next(it)
        hw = batch.hw.long()
        images += len(batch.ids)
        tokens += int((hw[:, 0] * hw[:, 1]).sum() + batch.text_len.sum())
    seconds = time.perf_counter() - start
    return {
        "workers": workers,
        "batches_per_s": BATCHES / seconds,
        "images_per_s": images / seconds,
        "joint_tokens_per_s": tokens / seconds,
        "seconds": seconds,
    }


def main() -> None:
    check_script(globals(), owners=("../../train/moss.py", "../../data"))
    cfg = data_config(globals(), num_workers=0)
    dataset = MossDataset(cfg)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"loader_{cfg.resolution_mode}_{cfg.batching}.json")
    records = []
    for workers in WORKERS:
        records.append(run(dataset, workers))
        print(json.dumps(records[-1]))
        with open(path, "w", encoding="utf-8") as handle:
            config = {"resolution": RESOLUTION_MODE, "batching": BATCHING}
            json.dump({"config": config, "records": records}, handle, indent=1)


if __name__ == "__main__":
    main()
