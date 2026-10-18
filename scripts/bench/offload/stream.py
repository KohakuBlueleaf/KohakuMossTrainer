"""Offload bench: copy bandwidth, and the step cost of weight streaming and activation
offload on the real model (`--weights`) and real data (`--datasets`), each case a short
LoRA trainer run (`entry.run`) in its own process and the bandwidth in one more (the
parent never touches the device); writes runs/bench/offload/<gpu>.json and every run's
directory (its checkpoints removed) under runs/bench/offload/runs/<gpu>/.

    python scripts/bench/offload/stream.py --weights <root>/base \
        --datasets '[{"source": "kohya", "path": "<dataset>"}]' --device cuda:0
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from kohakumoss.offload.accel import default_type
from kohakumoss.offload.accel import module as accel
from kohakumoss.training.entry import run
from kohakumoss.training.knobs import resolve_precision
from kohakumoss_bench.plan.knobs import script_defaults
from kohakumoss_bench.report.results import read_rows, training_summary

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "train" / "moss.py"
MB = 1024 * 1024
BLOCK = {"checkpoint": "block"}
CASES = {
    "resident": {},
    "checkpoint": BLOCK,
    "act_offload": {**BLOCK, "act_offload": True},
    "stream_pinned": {**BLOCK, "stream": "ring", "stream_resident": 2, "pinned": True},
    "stream_staged": {**BLOCK, "stream": "ring", "stream_resident": 2, "pinned": False},
}


def bandwidth(device: torch.device, sizes_mb=(4, 64, 256)) -> list[dict]:
    rows = []
    for size in sizes_mb:
        n = size * MB // 2
        gpu = torch.empty(n, dtype=torch.float16, device=device)
        for kind in ("pinned", "pageable"):
            host = torch.empty(n, dtype=torch.float16, pin_memory=kind == "pinned")
            for direction in ("h2d", "d2h"):
                accel(device).synchronize(device)
                t0 = time.perf_counter()
                for _ in range(5):
                    if direction == "h2d":
                        gpu.copy_(host, non_blocking=True)
                    else:
                        host.copy_(gpu, non_blocking=True)
                accel(device).synchronize(device)
                dt = (time.perf_counter() - t0) / 5
                rows.append(
                    {
                        "size_mb": size,
                        "host": kind,
                        "dir": direction,
                        "GBps": size / 1024 / dt,
                    }
                )
    return rows


def knobs(args, overrides: dict, out_dir: str) -> dict:
    """The train script's defaults, the bench's short LoRA run on the `speed` profile,
    then the case's memory overrides."""
    k = script_defaults(SCRIPT)
    k.update(
        WEIGHTS=args.weights,
        VARIANT=args.variant,
        ALGO="lora",
        DIM=16,
        ALPHA=16.0,
        PRECISION=args.precision,
        DATASETS=json.loads(args.datasets),
        LATENT_CACHE=args.cache,
        TEXT_CACHE=args.cache,
        CACHE_DIR=args.cache_dir,
        RESOLUTION_MODE="bucket",
        TARGET_AREA=args.area,
        BATCHING="count",
        BATCH_SIZE=args.batch,
        NUM_WORKERS=0,
        STEPS=args.warmup + args.iters,
        LOG_EVERY=1,
        METRICS_EVERY=1,
        PROGRESS=False,
        SAVE_EVERY=0,
        SAMPLE_EVERY=0,
        VALIDATE_EVERY=0,
        EXPORT_FORMATS=(),
        LOGGER="console",
        SCHEDULE={"mode": "constant"},
        MEMORY_PROFILE="speed",
        MEMORY_OVERRIDES=overrides,
        DEVICES=(torch.device(args.device).index or 0,),
        OUTPUT_DIR=out_dir,
        RUN_NAME="train",
    )
    k.update(json.loads(args.knobs) if args.knobs else {})
    k["PRECISION"] = resolve_precision(k)
    if k["PRECISION"] == "fp16":
        k.update(FP16_FIX="online", FP16_FIX_KWARGS={"c": 1 / 16})
    return k


def steady_ms(out_dir: str, warmup: int) -> list[float]:
    """Step times (ms) of the run's steps after `warmup`."""
    rows = read_rows(Path(out_dir) / "train" / "metrics.jsonl")
    return [1e3 * r["step_time"] for r in rows if r.get("step", 0) > warmup]


def step_case(args, name: str, gpu: str) -> dict:
    """One case's trainer run: step time (median, min, max after the warmup steps),
    tokens/s, allocator peaks, the run's nvidia-smi device peak and the GB streamed or
    offloaded per step from its metric rows."""
    out_dir = os.path.join(args.out, "runs", gpu, name)
    k = knobs(args, CASES[name], out_dir)
    try:
        run(k)
    except torch.OutOfMemoryError as err:
        return {"case": name, "gpu": gpu, "oom": True, "error": str(err)[:400]}
    finally:
        shutil.rmtree(Path(out_dir) / "train" / "checkpoints", ignore_errors=True)
    s = training_summary(Path(out_dir), warmup=args.warmup)
    step = s.get("step_time_median") or float("nan")
    times = steady_ms(out_dir, args.warmup) or [float("nan")]
    job = s.get("rank_summary", {}).get("job", {})
    return {
        "case": name,
        "gpu": gpu,
        "precision": k["PRECISION"],
        "overrides": CASES[name],
        "ms": 1e3 * step,
        "ms_min": min(times),
        "ms_max": max(times),
        "steady_steps": s.get("steady_steps"),
        "tokens_per_s": s.get("tokens_per_s_median"),
        "peak_alloc_GiB": s.get("vram_peak_alloc_max"),
        "peak_reserved_GiB": s.get("vram_peak_reserved_max"),
        "device_used_GiB": s.get("vram_device_max"),
        "smi_peak_GiB": job.get("smi_mem_peak_gib"),
        "h2d_GB_per_step": s.get("xfer_h2d_gb_median") or 0.0,
        "d2h_GB_per_step": s.get("xfer_d2h_gb_median") or 0.0,
        "run_dir": out_dir,
        "summary": s,
    }


def child(args: list[str]) -> dict | None:
    """The ROW of this script run as a child process with `args`, else the tail of its
    output as {"error": ...}."""
    out = subprocess.run(
        [sys.executable, __file__, *args], capture_output=True, text=True
    )
    found = [line[4:] for line in out.stdout.splitlines() if line.startswith("ROW ")]
    return (
        json.loads(found[0]) if found else {"error": (out.stdout + out.stderr)[-2000:]}
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--datasets", required=True, help="DATASETS (JSON list)")
    ap.add_argument("--device", default=f"{default_type()}:0")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--area", type=int, default=1024 * 1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3, help="steps left out of the stats")
    ap.add_argument("--iters", type=int, default=12, help="measured steps")
    ap.add_argument("--cache", default="offline", help="LATENT_CACHE / TEXT_CACHE")
    ap.add_argument("--cache-dir", default="runs/bench/offload/cache")
    ap.add_argument("--knobs", default="", help="JSON of extra train knobs")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--only", default="")
    ap.add_argument("--measure", action="store_true", help="child: the bandwidth row")
    ap.add_argument("--out", default="runs/bench/offload")
    args, _ = ap.parse_known_args()
    if args.only or args.measure:
        device = torch.device(args.device)
        accel(device).set_device(device)
        name = accel(device).get_device_name(device)
        gpu = name.replace(" ", "_").replace("/", "_")
        row = (
            {"gpu": gpu, "name": name, "bandwidth": bandwidth(device)}
            if args.measure
            else step_case(args, args.only, gpu)
        )
        print("ROW " + json.dumps(row))
        return
    measured = child(["--measure"] + sys.argv[1:])
    steps = [
        {"case": name, **child(["--only", name] + sys.argv[1:])}
        for name in args.cases.split(",")
    ]
    gpu = measured.get("gpu") or next((s["gpu"] for s in steps if "gpu" in s), "gpu")
    report = {
        "gpu": measured.get("name", gpu),
        "args": vars(args),
        "bandwidth": measured.get("bandwidth", []),
        "steps": steps,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{gpu}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for row in report["bandwidth"]:
        print(
            f"bw {row['size_mb']:>4}MB {row['host']:>8} {row['dir']} {row['GBps']:6.2f} GB/s"
        )
    for row in report["steps"]:
        print(
            " ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()
                if k != "summary"
            )
        )
    print("wrote", path)


if __name__ == "__main__":
    main()
