"""Hardware-limit check of memory-knob combinations on the real model (`--weights`) and
real data (`--datasets`), each case a short trainer run (`entry.run`) in its own
process and the bandwidth / MMA / host-AdamW rates in one more (the parent never
touches the device); writes runs/bench/offload/limits_<gpu>.json and every run's
directory (its checkpoints removed) under runs/bench/offload/runs/<gpu>/. See
docs/internals/offload.md.

    python scripts/bench/offload/limits.py --weights <root>/base \
        --datasets '[{"source": "kohya", "path": "<dataset>"}]' --device cuda:0
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from kohakumoss.offload.accel import default_type
from kohakumoss.offload.accel import module as accel
from kohakumoss.offload.optim.cpu_optim import CPUAdamW, adamw_host
from kohakumoss.training import entry
from kohakumoss.training.knobs import resolve_precision
from kohakumoss_bench.plan.knobs import script_defaults
from kohakumoss_bench.report.results import read_rows, training_summary
from kohakumoss_kernels.hardware.arch import arch_of

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "train" / "moss.py"
MB = 1024 * 1024
CASES = {
    "full/resident/gpu": dict(stream="off", optim_state="gpu"),
    "full/resident/gpu/acc4": dict(stream="off", optim_state="gpu", acc=4),
    "full/grad_offload/gpu/acc4": dict(
        stream="off", optim_state="gpu", grad_offload=True, acc=4
    ),
    "full/stream_states": dict(stream="off", optim_state="stream", fused_step=True),
    "full/cpu_adamw": dict(stream="off", optim_state="cpu"),
    "full/swap/gpu_resident_states": dict(stream="swap", optim_state="gpu"),
    "full/swap/gpu_streamed_states": dict(stream="swap", optim_state="stream"),
    "full/swap/gpu_streamed_states/acc4": dict(
        stream="swap", optim_state="stream", acc=4
    ),
    "full/swap/cpu_adamw": dict(stream="swap", optim_state="cpu"),
    "adapter/resident": dict(stream="off", optim_state="gpu", full=False),
    "adapter/act_offload": dict(
        stream="off", optim_state="gpu", act_offload=True, full=False
    ),
    "adapter/ring": dict(stream="ring", optim_state="gpu", full=False),
    "adapter/ring/act_offload/grad_offload": dict(
        stream="ring",
        optim_state="gpu",
        act_offload=True,
        grad_offload=True,
        full=False,
    ),
    "adapter/ring/act_offload": dict(
        stream="ring", optim_state="gpu", act_offload=True, full=False
    ),
    "adapter/ring/grad_offload": dict(
        stream="ring", optim_state="gpu", grad_offload=True, full=False
    ),
    "adapter/ring/int8": dict(
        stream="ring", optim_state="gpu", base="int8", full=False
    ),
}
RESIDENT = ("full/resident/gpu", "full/resident/gpu/acc4", "adapter/resident")


def bandwidth(device) -> dict[str, float]:
    n = 256 * MB // 2
    gpu = torch.empty(n, dtype=torch.float16, device=device)
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)
    out = {}
    for name, fn in (
        ("h2d", lambda: gpu.copy_(host, non_blocking=True)),
        ("d2h", lambda: host.copy_(gpu, non_blocking=True)),
    ):
        fn()
        accel(device).synchronize(device)
        t0 = time.perf_counter()
        for _ in range(5):
            fn()
        accel(device).synchronize(device)
        out[name] = 5 * 256 * MB / (time.perf_counter() - t0) / 1e9
    gpu2 = torch.empty_like(gpu)
    host2 = torch.empty(n, dtype=torch.float16, pin_memory=True)
    s1, s2 = accel(device).Stream(device), accel(device).Stream(device)
    accel(device).synchronize(device)
    t0 = time.perf_counter()
    for _ in range(5):
        with accel(device).stream(s1):
            gpu.copy_(host, non_blocking=True)
        with accel(device).stream(s2):
            host2.copy_(gpu2, non_blocking=True)
    accel(device).synchronize(device)
    out["bidir_total"] = 2 * 5 * 256 * MB / (time.perf_counter() - t0) / 1e9
    return out


def mma_peak(device, dtype) -> float:
    a = torch.randn(8192, 8192, device=device, dtype=dtype)
    b = torch.randn(8192, 8192, device=device, dtype=dtype)
    a @ b
    accel(device).synchronize(device)
    t0 = time.perf_counter()
    for _ in range(5):
        a @ b
    accel(device).synchronize(device)
    return 5 * 2 * 8192**3 / (time.perf_counter() - t0) / 1e12


def host_adamw_rate(numel: int = 64 * MB) -> float:
    """Parameters per second of one host AdamW update (`adamw_host`, fp32)."""
    t = [torch.randn(numel) for _ in range(4)]
    step = torch.zeros(())
    adamw_host(*t, step, 1e-4, 0.9, 0.999, 1e-8, 0.01)
    t0 = time.perf_counter()
    for _ in range(3):
        adamw_host(*t, step, 1e-4, 0.9, 0.999, 1e-8, 0.01)
    return 3 * numel / (time.perf_counter() - t0)


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


def case_knobs(name: str) -> tuple[dict, int, bool]:
    """(train knobs, GRAD_ACC, full finetune) of case `name`."""
    spec = dict(CASES[name])
    acc = spec.pop("acc", 1)
    full = spec.pop("full", True)
    base = spec.pop("base", "compute")
    overrides = dict(checkpoint="block", stream_resident=2, base_precision=base, **spec)
    case = dict(
        ALGO="full" if full else "lora", GRAD_ACC=acc, MEMORY_OVERRIDES=overrides
    )
    return case, acc, full


def host_adamw_params(bundle) -> int:
    """Parameters the bundle's host AdamW updates (0 without one)."""
    opts = [getattr(bundle, name, None) for name in ("optimizer", "shared")]
    return sum(
        p.numel()
        for opt in opts
        if isinstance(opt, CPUAdamW)
        for g in opt.param_groups
        for p in g["params"]
    )


def steady_ms(out_dir: str, warmup: int) -> list[float]:
    """Step times (ms) of the run's steps after `warmup`."""
    rows = read_rows(Path(out_dir) / "train" / "metrics.jsonl")
    return [1e3 * r["step_time"] for r in rows if r.get("step", 0) > warmup]


def measure(device) -> dict:
    """The device's name, copy bandwidth, MMA peak and the host AdamW rate."""
    name = accel(device).get_device_name(device)
    dtype = torch.bfloat16 if arch_of(device).has_bf16 else torch.float16
    return {
        "gpu": name.replace(" ", "_").replace("/", "_"),
        "name": name,
        "mma_dtype": str(dtype),
        "bandwidth": bandwidth(device),
        "mma": mma_peak(device, dtype),
        "host_rate": host_adamw_rate(),
    }


def child(args: list[str]) -> dict:
    """The ROW of this script run as a child process with `args`, else the tail of its
    output as {"error": ...}."""
    out = subprocess.run(
        [sys.executable, __file__, *args], capture_output=True, text=True
    )
    found = [line[4:] for line in out.stdout.splitlines() if line.startswith("ROW ")]
    return (
        json.loads(found[0]) if found else {"error": (out.stdout + out.stderr)[-2000:]}
    )


def run_case(args, name: str, gpu: str) -> dict:
    """One case's trainer run: step time (median, min, max after the warmup steps),
    copies, allocator peaks, the run's nvidia-smi device peak and model FLOPs per step
    from its metric rows."""
    case, acc, full = case_knobs(name)
    out_dir = os.path.join(args.out, "runs", gpu, name.replace("/", "_"))
    k = knobs(args, case, out_dir)
    held = {}
    build = entry.build_optim

    def capture(*a, **kw):
        out = build(*a, **kw)
        held["bundle"] = out[0]
        return out

    entry.build_optim = capture
    try:
        entry.run(k)
    except torch.OutOfMemoryError as err:
        return {"case": name, "oom": True, "error": str(err)[:400], "knobs": case}
    finally:
        shutil.rmtree(Path(out_dir) / "train" / "checkpoints", ignore_errors=True)
    s = training_summary(Path(out_dir), warmup=args.warmup)
    step = s.get("step_time_median") or float("nan")
    times = steady_ms(out_dir, args.warmup) or [float("nan")]
    job = s.get("rank_summary", {}).get("job", {})
    return {
        "case": name,
        "precision": k["PRECISION"],
        "knobs": case,
        "ms": 1e3 * step,
        "ms_min": min(times),
        "ms_max": max(times),
        "steady_steps": s.get("steady_steps"),
        "peak_alloc_GiB": s.get("vram_peak_alloc_max"),
        "peak_reserved_GiB": s.get("vram_peak_reserved_max"),
        "device_used_GiB": s.get("vram_device_max"),
        "smi_peak_GiB": job.get("smi_mem_peak_gib"),
        "h2d_GB": s.get("xfer_h2d_gb_median") or 0.0,
        "d2h_GB": s.get("xfer_d2h_gb_median") or 0.0,
        "flops": (s.get("tflops_median") or float("nan")) * 1e12 * step,
        "acc": acc,
        "full": full,
        "host_adamw_params": host_adamw_params(held.get("bundle")),
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
    ap.add_argument("--warmup", type=int, default=3, help="steps left out of the stats")
    ap.add_argument("--iters", type=int, default=12, help="measured steps")
    ap.add_argument("--cache", default="offline", help="LATENT_CACHE / TEXT_CACHE")
    ap.add_argument("--cache-dir", default="runs/bench/offload/cache")
    ap.add_argument("--knobs", default="", help="JSON of extra train knobs")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--only", default="")
    ap.add_argument("--measure", action="store_true", help="child: the rates row")
    ap.add_argument("--out", default="runs/bench/offload")
    args, _ = ap.parse_known_args()
    if args.only or args.measure:
        device = torch.device(args.device)
        accel(device).set_device(device)
        gpu = accel(device).get_device_name(device).replace(" ", "_").replace("/", "_")
        row = measure(device) if args.measure else run_case(args, args.only, gpu)
        print("ROW " + json.dumps(row))
        return
    rates = child(["--measure"] + sys.argv[1:])
    if "error" in rates:
        raise SystemExit("rate measurement failed:\n" + rates["error"])
    gpu, bw, peak, host_rate = (
        rates["gpu"],
        rates["bandwidth"],
        rates["mma"],
        rates["host_rate"],
    )
    rows = []
    for name in args.cases.split(","):
        rows.append({"case": name, **child(["--only", name] + sys.argv[1:])})
        if "error" in rows[-1]:
            print(json.dumps(rows[-1]))
    measured = [r for r in rows if "ms" in r]
    resident = {(r["full"], r["acc"]): r for r in measured if r["case"] in RESIDENT}
    for r in measured:
        copy_ms = 1e3 * max(
            r["h2d_GB"] / bw["h2d"],
            r["d2h_GB"] / bw["d2h"],
            (r["h2d_GB"] + r["d2h_GB"]) / bw["bidir_total"],
        )
        base = resident.get((r["full"], r["acc"])) or resident.get((r["full"], 1))
        compute_ms = base["ms"] * r["acc"] / base["acc"] if base else float("nan")
        r["copy_bound_ms"] = copy_ms
        r["compute_ms"] = compute_ms
        r["flop_bound_ms"] = 1e3 * r["flops"] / (peak * 1e12)
        r["host_adamw_ms"] = 1e3 * r["host_adamw_params"] / host_rate
        work = compute_ms if math.isfinite(compute_ms) else r["flop_bound_ms"]
        r["overlap"] = r["ms"] / max(copy_ms, work, r["host_adamw_ms"])
        print(
            f"{r['case']:<42} ms={r['ms']:8.1f} ({r['ms_min']:.0f}-{r['ms_max']:.0f}) "
            f"copy={copy_ms:8.1f} compute={compute_ms:8.1f} "
            f"host_adamw={r['host_adamw_ms']:8.1f} "
            f"flop_bound={r['flop_bound_ms']:7.1f} overlap={r['overlap']:5.2f} "
            f"h2d={r['h2d_GB']:6.2f}GB d2h={r['d2h_GB']:6.2f}GB "
            f"peak={r['peak_alloc_GiB'] or 0.0:5.2f}GiB smi={r['smi_peak_GiB'] or 0.0:5.2f}GiB"
        )
    report = {
        "gpu": rates["name"],
        "mma_dtype": rates["mma_dtype"],
        "bandwidth_GBps": bw,
        "mma_TFLOPs": peak,
        "host_adamw_params_per_s": host_rate,
        "args": vars(args),
        "rows": rows,
    }
    print("bandwidth", bw, "mma TFLOP/s", round(peak, 1))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"limits_{gpu}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
