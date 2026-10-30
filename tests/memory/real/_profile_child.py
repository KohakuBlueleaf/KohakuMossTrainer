"""Child process of test_profile_memory: a short real-weights LoRA training run under one
memory profile (argv 1) on a slice of the KOHAKUMOSS_DATASET dataset (argv 3), writing the trainer's
metric rows under argv 2; argv 4 "broken" runs the activation offload without its host
pacing and removes the host workers' stop sentinel (the negatives of the reserved-memory
and thread checks). Prints {"rows": path, "threads": host worker threads left after the
run} or {"oom": true} as JSON."""

import gc
import json
import os
import sys
import threading
from pathlib import Path

import torch

from kohakumoss.cli.config.validate import script_defaults
from kohakumoss.offload import worker
from kohakumoss.offload.activation import ActivationOffload
from kohakumoss.training.entry import run
from kohakumoss.training.knobs import resolve_precision
from tests._util.dataset import kohya_source

ROOT = Path(__file__).resolve().parents[3]
STEPS = 6


def knobs(profile: str, out: Path, dataset: Path, tag: str) -> dict:
    k = script_defaults(ROOT, "scripts/train/moss.py")
    k.update(
        WEIGHTS=str(Path(os.environ["KOHAKUMOSS_WEIGHTS"]) / "base"),
        VARIANT="base",
        ALGO="lora",
        DIM=8,
        ALPHA=8.0,
        DATASETS=[kohya_source(dataset)],
        LATENT_CACHE="off",
        TEXT_CACHE="off",
        TEXT_ENCODE={"memory": "stream"},
        MEMORY_PROFILE=profile,
        PRECISION="auto",
        DEVICES=(0,),
        STEPS=STEPS,
        SCHEDULE={"mode": "constant", "warmup": 0, "end": -1},
        TUNE="off",
        SAMPLE_EVERY=0,
        SAVE_EVERY=0,
        EXPORT_FORMATS=(),
        LOG_EVERY=1,
        METRICS_EVERY=1,
        METRICS_LEVEL="debug",
        PROGRESS=False,
        NUM_WORKERS=0,
        OUTPUT_DIR=str(out),
        RUN_NAME=f"{profile}-{tag}",
    )
    k["PRECISION"] = resolve_precision(k)
    if k["PRECISION"] == "fp16":
        k.update(FP16_FIX="online", FP16_FIX_KWARGS={"c": 1 / 16})
    return k


def _no_sentinel(jobs, thread, timeout) -> None:
    return None


def _unpaced(self, kind, block, ev) -> None:
    return None


def _workers() -> int:
    return sum(t.name.startswith("kohakumoss-") for t in threading.enumerate())


def main() -> None:
    profile, out, dataset = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    broken = sys.argv[4:] == ["broken"]
    if broken:
        ActivationOffload._pace = _unpaced
        worker._stop = _no_sentinel
    k = knobs(profile, out, dataset, "broken" if broken else "run")
    before = _workers()
    try:
        run(k)
    except torch.OutOfMemoryError:
        print(json.dumps({"oom": True}))
        return
    gc.collect()
    rows = out / k["RUN_NAME"] / "metrics.jsonl"
    print(json.dumps({"rows": str(rows), "threads": _workers() - before}))


if __name__ == "__main__":
    main()
