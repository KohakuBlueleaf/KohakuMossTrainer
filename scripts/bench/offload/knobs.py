"""One measured line per VRAM knob, each in isolation, on the real model (`--weights`)
and real data (`--datasets`), each case a short trainer run (`entry.run`) in its own
process (the parent never touches the device); writes runs/bench/offload/knobs_<gpu>.json
and every run's directory (log,
metric rows; its checkpoints removed) under runs/bench/offload/runs/<gpu>/. See
docs/internals/offload.md.

    python scripts/bench/offload/knobs.py --weights <root>/base \
        --datasets '[{"source": "kohya", "path": "<dataset>"}]' --device cuda:0
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from kohakumoss.offload.accel import default_type
from kohakumoss.offload.accel import module as accel
from kohakumoss.training.entry import run
from kohakumoss.training.knobs import resolve_precision
from kohakumoss_bench.plan.knobs import script_defaults
from kohakumoss_bench.report.results import read_rows, training_summary

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "train" / "moss.py"
FULL_FT = {
    "gpu_adamw": dict(optim_state="gpu"),
    "cpu_adamw": dict(optim_state="cpu"),
    "grad_offload_cpu_adamw": dict(optim_state="cpu", grad_offload=True),
    "grad_offload_gpu_adamw": dict(optim_state="gpu", grad_offload=True),
}
AGGRESSIVE = dict(stream="swap", stream_resident=2, grad_offload=True)


def knobs(args, case: dict, out_dir: str) -> dict:
    """The train script's defaults, the bench's short run, then the case's knobs."""
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
        GRAD_ACC=args.accum,
        NUM_WORKERS=0,
        STEPS=args.warmup + args.iters,
        LOG_EVERY=1,
        METRICS_EVERY=1,
        METRICS_LEVEL="debug",
        PROGRESS=False,
        SAVE_EVERY=0,
        SAMPLE_EVERY=0,
        VALIDATE_EVERY=0,
        EXPORT_FORMATS=(),
        LOGGER="console",
        SCHEDULE={"mode": "constant"},
        MEMORY_PROFILE="balanced",
        DEVICES=(torch.device(args.device).index or 0,),
        OUTPUT_DIR=out_dir,
        RUN_NAME="train",
    )
    k.update(case)
    k.update(json.loads(args.knobs) if args.knobs else {})
    k["PRECISION"] = resolve_precision(k)
    if k["PRECISION"] == "fp16":
        k.update(FP16_FIX="online", FP16_FIX_KWARGS={"c": 1 / 16})
    return k


def full_ft(args, mode: str) -> tuple[str, dict]:
    overrides = {"checkpoint": "block", "stream": "off", **FULL_FT[mode]}
    return f"full_ft/{mode}/acc{args.accum}", dict(
        ALGO="full", MEMORY_OVERRIDES=overrides
    )


def granularity(args, value: str) -> tuple[str, dict]:
    return f"checkpoint/{value}/acc{args.accum}", dict(
        MEMORY_PROFILE="speed", MEMORY_OVERRIDES={"checkpoint": value}
    )


def aggressive(args, batch: int) -> tuple[str, dict]:
    overrides = {"checkpoint": "block", **AGGRESSIVE, "optim_state": "stream"}
    name = f"swap+grad_offload+stream_states/batch{batch}/acc{args.accum}"
    return name, dict(ALGO="full", BATCH_SIZE=batch, MEMORY_OVERRIDES=overrides)


CASES = {
    "full_ft:gpu_adamw": (full_ft, "gpu_adamw"),
    "full_ft:cpu_adamw": (full_ft, "cpu_adamw"),
    "full_ft:grad_offload_cpu_adamw": (full_ft, "grad_offload_cpu_adamw"),
    "full_ft:grad_offload_gpu_adamw": (full_ft, "grad_offload_gpu_adamw"),
    "ckpt:none": (granularity, "none"),
    "ckpt:block": (granularity, "block"),
    "ckpt:sub-block": (granularity, "sub-block"),
    "ckpt:selective": (granularity, "selective"),
    "ckpt:partial:0.5": (granularity, "partial:0.5"),
    "aggressive:batch": (aggressive, "batch"),
    "aggressive:big_batch": (aggressive, "big_batch"),
}


def steady_ms(out_dir: str, warmup: int) -> list[float]:
    """Step times (ms) of the run's steps after `warmup`."""
    rows = read_rows(Path(out_dir) / "train" / "metrics.jsonl")
    return [1e3 * r["step_time"] for r in rows if r.get("step", 0) > warmup]


def run_case(args, label: str, gpu: str) -> dict:
    """One case's trainer run: the summary of its metric rows (medians after the warmup
    steps, maxima over the run), step-time min / max, the run's nvidia-smi device peak
    and the knobs it ran with."""
    fn, arg = CASES[label]
    arg = {"batch": args.batch, "big_batch": args.big_batch}.get(arg, arg)
    name, case = fn(args, arg)
    out_dir = os.path.join(args.out, "runs", gpu, label.replace(":", "_"))
    k = knobs(args, case, out_dir)
    try:
        run(k)
    except torch.OutOfMemoryError as err:
        return {
            "case": name,
            "gpu": gpu,
            "oom": True,
            "error": str(err)[:400],
            "knobs": case,
        }
    finally:
        shutil.rmtree(Path(out_dir) / "train" / "checkpoints", ignore_errors=True)
    s = training_summary(Path(out_dir), warmup=args.warmup)
    step = s.get("step_time_median") or float("nan")
    tokens = (s.get("tokens_per_s_median") or float("nan")) * step
    times = steady_ms(out_dir, args.warmup) or [float("nan")]
    job = s.get("rank_summary", {}).get("job", {})
    return {
        "case": name,
        "gpu": gpu,
        "precision": k["PRECISION"],
        "knobs": case,
        "ms": step * 1e3,
        "ms_min": min(times),
        "ms_max": max(times),
        "steady_steps": s.get("steady_steps"),
        "tokens_per_step": tokens,
        "ms_per_1k_tokens": step * 1e6 / tokens,
        "peak_alloc_GiB": s.get("vram_peak_alloc_max"),
        "peak_reserved_GiB": s.get("vram_peak_reserved_max"),
        "device_used_GiB": s.get("vram_device_max"),
        "smi_peak_GiB": job.get("smi_mem_peak_gib"),
        "h2d_GB_per_step": s.get("xfer_h2d_gb_median") or 0.0,
        "d2h_GB_per_step": s.get("xfer_d2h_gb_median") or 0.0,
        "run_dir": out_dir,
        "summary": s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--datasets", required=True, help="DATASETS (JSON list)")
    ap.add_argument("--device", default=f"{default_type()}:0")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--area", type=int, default=1024 * 1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--big-batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3, help="steps left out of the stats")
    ap.add_argument("--iters", type=int, default=12, help="measured steps")
    ap.add_argument("--cache", default="offline", help="LATENT_CACHE / TEXT_CACHE")
    ap.add_argument("--cache-dir", default="runs/bench/offload/cache")
    ap.add_argument("--knobs", default="", help="JSON of extra train knobs")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="runs/bench/offload")
    args, _ = ap.parse_known_args()
    if args.only:
        device = torch.device(args.device)
        dm = accel(device)
        dm.set_device(device)
        gpu = dm.get_device_name(device).replace(" ", "_").replace("/", "_")
        print("ROW " + json.dumps(run_case(args, args.only, gpu)))
        return
    rows = []
    for label in args.cases.split(","):
        cmd = [sys.executable, __file__, "--only", label] + sys.argv[1:]
        out = subprocess.run(cmd, capture_output=True, text=True)
        found = [
            line[4:] for line in out.stdout.splitlines() if line.startswith("ROW ")
        ]
        rows.append(
            json.loads(found[0])
            if found
            else {"case": label, "error": (out.stdout + out.stderr)[-2000:]}
        )
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "summary"}))
    gpu = next((r["gpu"] for r in rows if "gpu" in r), "gpu")
    report = {"gpu": gpu.replace("_", " "), "args": vars(args), "rows": rows}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"knobs_{gpu}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
