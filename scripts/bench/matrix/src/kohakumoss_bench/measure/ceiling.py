"""Empirical tensor ceiling of the visible card 0: square n x n x n `torch.mm` in fp16 and
bf16 (PyTorch's default fp32 accumulation, with reduced-precision reduction allowed and
disallowed), warm, the median of `repeats` timed launches, with the SM clock, power and
temperature nvidia-smi samples over the end of the warm loop and the timed launches, and
the launched kernel names. JSON to --out.
See docs/performance/gemm-efficiency.md.

    python -m kohakumoss_bench.measure.ceiling --out ceiling.json [--n 8192]
"""

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

SMI_PERIOD_MS = 20
SMI_STAMP = "%Y/%m/%d %H:%M:%S.%f"


def window_samples(lines: list[str], lo: datetime, hi: datetime) -> list[list[float]]:
    """[clock MHz, power W, temperature C] of the nvidia-smi rows (timestamp first)
    stamped within [lo, hi]; rows that do not parse are skipped."""
    out = []
    for line in lines:
        cells = [c.strip() for c in line.split(",")]
        try:
            at = datetime.strptime(cells[0], SMI_STAMP)
            values = [float(c) for c in cells[1:4]]
        except (ValueError, IndexError):
            continue
        if lo <= at <= hi and len(values) == 3:
            out.append(values)
    return out


def kernel_names(a: torch.Tensor, b: torch.Tensor) -> list[str]:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.mm(a, b)
        torch.cuda.synchronize()
    names = {e.key for e in prof.key_averages() if e.device_type.name == "CUDA"}
    return sorted(n for n in names if not n.startswith("Memset"))


def measure(
    n: int,
    dtype: torch.dtype,
    reduced: bool,
    repeats: int,
    warm_s: float,
    window_s: float,
) -> dict:
    """One configuration: timed launches, the clock window, TFLOP/s and FLOP/clk/SM."""
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced
    props = torch.cuda.get_device_properties(0)
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    smi = subprocess.Popen(
        [
            "nvidia-smi",
            "-i",
            f"GPU-{props.uuid}",
            "--query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            "-lms",
            str(SMI_PERIOD_MS),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    end = time.perf_counter() + warm_s
    while time.perf_counter() < end:
        for _ in range(4):
            torch.mm(a, b)
        torch.cuda.synchronize()
    lo = datetime.now() - timedelta(seconds=window_s)
    ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(a, b)
        stop.record()
        stop.synchronize()
        ms.append(start.elapsed_time(stop))
    hi = datetime.now()
    time.sleep(0.2)
    smi.terminate()
    samples = window_samples(smi.communicate()[0].splitlines(), lo, hi)
    median_ms = statistics.median(ms)
    flop = 2.0 * n**3
    mhz = statistics.median(s[0] for s in samples) if samples else None
    per_clk = (
        flop / (median_ms * 1e-3) / (props.multi_processor_count * mhz * 1e6)
        if mhz
        else None
    )
    return {
        "n": n,
        "dtype": str(dtype).removeprefix("torch."),
        "reduced_reduction": reduced,
        "median_ms": median_ms,
        "min_ms": min(ms),
        "max_ms": max(ms),
        "tflops": flop / (median_ms * 1e-3) / 1e12,
        "tflops_best": flop / (min(ms) * 1e-3) / 1e12,
        "clock_mhz": mhz,
        "clock_mhz_min": min((s[0] for s in samples), default=None),
        "clock_mhz_max": max((s[0] for s in samples), default=None),
        "power_w": statistics.median(s[1] for s in samples) if samples else None,
        "temp_c_max": max((s[2] for s in samples), default=None),
        "clock_samples": len(samples),
        "flop_per_clk_per_sm": per_clk,
        "kernels": kernel_names(a, b),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warm-s", type=float, default=3.0)
    ap.add_argument(
        "--window-s",
        type=float,
        default=1.5,
        help="clock window before the timed launches",
    )
    args = ap.parse_args(argv)
    props = torch.cuda.get_device_properties(0)
    res = {
        "device": props.name,
        "sm": props.major * 10 + props.minor,
        "sms": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "runs": [],
    }
    for dtype in (torch.float16, torch.bfloat16):
        for reduced in (True, False):
            r = measure(
                args.n, dtype, reduced, args.repeats, args.warm_s, args.window_s
            )
            res["runs"].append(r)
            print(
                f"{r['dtype']:9} reduced={reduced!s:5} {r['median_ms']:.3f} ms "
                f"{r['tflops']:.1f} TFLOP/s at {r['clock_mhz']} MHz, "
                f"{r['power_w']} W -> {r['flop_per_clk_per_sm']} FLOP/clk/SM; "
                f"{r['kernels']}",
                flush=True,
            )
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
