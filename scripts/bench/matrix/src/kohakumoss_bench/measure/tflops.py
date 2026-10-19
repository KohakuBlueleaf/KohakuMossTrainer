"""Achieved tensor TFLOP/s per GEMM / attention kernel of one training run's profiler
window: ncu application replay of the launches whose name matches a regex, with the HMMA
warp-instruction count, the fp16 -> fp32 tensor-op count and the SM clock of each launch,
against the peak SMs x FLOP/clk/SM x that launch's clock.
See docs/performance/gemm-efficiency.md.

    python -m kohakumoss_bench.measure.tflops run --config <train config> --out <dir> \
        --sm 86 --sms 82 [--ncu "sudo -n ncu"] [--env-file <json>]
    python -m kohakumoss_bench.measure.tflops report --report <dir>/tf.ncu-rep \
        --sm 86 --sms 82 [--out <dir>]
"""

import argparse
import csv
import io
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

from kohakumoss_bench.measure import ncu

DURATION = "gpu__time_duration.sum"
CLOCK = "sm__cycles_elapsed.avg.per_second"
INST = {
    75: "sm__inst_executed_pipe_tensor_op_hmma.sum",
    86: "sm__inst_executed_pipe_tensor_op_hmma.sum",
    120: "sm__inst_executed_pipe_tensor_subpipe_hmma_op_hmma.sum",
}
OPS = {
    75: "sm__ops_path_tensor_src_fp16_bf16_tf32_dst_fp32.sum",
    86: "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
    120: "sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32.sum",
}
# FLOP of one HMMA warp instruction: m16n8k16 / m16n8k8, and the arch's fp16 shape
SHAPES = (("16816", 4096), ("1688", 2048))
ARCH_FLOP_PER_INST = {75: 2048, 86: 4096, 89: 4096, 120: 4096}
# dense fp16-input / fp32-accumulate FLOP per clock per SM
FLOP_PER_CLK = {70: 1024, 75: 512, 86: 512, 89: 512, 120: 512}
REGEX = "gemm|flash"
REPORT = "tf"


def flop_per_inst(name: str, sm: int, override: int | None = None) -> int:
    """FLOP of one HMMA warp instruction of kernel `name` on `sm`: `override`, else the
    mma shape in the name, else the arch's fp16 shape; ValueError for an arch without
    one."""
    if override:
        return override
    for token, flop in SHAPES:
        if token in name:
            return flop
    if sm not in ARCH_FLOP_PER_INST:
        raise ValueError(f"no HMMA shape for sm{sm}: pass --flop-per-inst")
    return ARCH_FLOP_PER_INST[sm]


def kind(name: str) -> str:
    """The kind of kernel `name`: "attention-fwd" / "attention-bwd" (flash) or "gemm"."""
    low = name.lower()
    if "flash" not in low:
        return "gemm"
    return "attention-bwd" if "bwd" in low else "attention-fwd"


def metrics(inst: str, ops: str) -> list[str]:
    return [DURATION, CLOCK, inst, ops, "launch__grid_size", "launch__block_size"]


def command(
    tool: list[str], inst: str, ops: str, out: Path, target: list[str], regex: str
) -> list[str]:
    """ncu application replay over `target`'s profiler window, only the launches whose
    demangled name matches `regex`, clocks and caches left alone, report `<out>.ncu-rep`.
    """
    return [
        *tool,
        *ncu.REPLAYS["application"],
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        f"regex:{regex}",
        "--target-processes",
        "all",
        "--clock-control",
        "none",
        "--cache-control",
        "none",
        "--metrics",
        ",".join(metrics(inst, ops)),
        "--export",
        str(out),
        "--force-overwrite",
        *target,
    ]


def _number(text: str | None) -> float | None:
    try:
        return float((text or "").replace(",", ""))
    except ValueError:
        return None


def launches(
    text: str,
    sm: int,
    sms: int,
    inst: str,
    ops: str,
    per_clk: float,
    flop_override: int | None = None,
) -> list[dict]:
    """One row per launch of `ncu --csv --page raw --print-units base` output (ns, Hz):
    FLOP from the instruction count, the tensor-op count, TFLOP/s of each, and the peak
    at the launch's clock."""
    lines = text.splitlines()
    start = next((i for i, s in enumerate(lines) if f'"{DURATION}"' in s), None)
    if start is None:
        return []
    table = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    head, body = table[0], table[2:]
    rows = []
    for cells in body:
        r = dict(zip(head, cells))
        ns, hz = _number(r.get(DURATION)), _number(r.get(CLOCK))
        if not ns or not hz:
            continue
        name = r.get("Kernel Name", "?")
        count, op_count = _number(r.get(inst)) or 0.0, _number(r.get(ops)) or 0.0
        flop = count * flop_per_inst(name, sm, flop_override)
        peak = sms * per_clk * hz
        rows.append(
            {
                "kernel": ncu.short(name),
                "kind": kind(name),
                "ns": ns,
                "clock_hz": hz,
                "flop": flop,
                "ops": op_count,
                "peak_flops": peak,
                "tflops": flop / ns / 1e3,
                "tflops_ops": op_count / ns / 1e3,
                "pct_peak": flop / (ns * 1e-9) / peak * 100,
                "grid": r.get("launch__grid_size"),
                "block": r.get("launch__block_size"),
            }
        )
    return rows


def _total(label: str, group: list[dict], whole_ns: float) -> dict:
    ns = sum(r["ns"] for r in group)
    flop = sum(r["flop"] for r in group)
    peak = sum(r["peak_flops"] * r["ns"] for r in group) / ns
    return {
        "kernel": label,
        "kind": group[0]["kind"],
        "launches": len(group),
        "ms": ns / 1e6,
        "share_pct": ns / whole_ns * 100,
        "tflops": flop / ns / 1e3,
        "tflops_ops": sum(r["ops"] for r in group) / ns / 1e3,
        "pct_peak": flop / (ns * 1e-9) / peak * 100,
        "peak_tflops": peak / 1e12,
        "clock_ghz": sum(r["clock_hz"] * r["ns"] for r in group) / ns / 1e9,
    }


def aggregate(rows: list[dict]) -> dict:
    """Per kernel name (FLOP-weighted TFLOP/s, time-weighted clock and peak, by time),
    per kind and overall, and the tensor-op / instruction-FLOP ratio over the launches
    (1.0 when both counters agree)."""
    if not rows:
        return {"kernels": [], "totals": [], "ops_ratio": None}
    whole = sum(r["ns"] for r in rows)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["kernel"], []).append(r)
    kernels = sorted(
        (_total(name, group, whole) for name, group in by.items()),
        key=lambda k: -k["ms"],
    )
    kinds: dict[str, list[dict]] = {}
    for r in rows:
        kinds.setdefault(r["kind"], []).append(r)
    totals = [_total(f"total {k}", g, whole) for k, g in sorted(kinds.items())]
    totals.append({**_total("total", rows, whole), "kind": "all"})
    ratios = [r["ops"] / r["flop"] for r in rows if r["ops"] and r["flop"]]
    ratio = {
        "median": statistics.median(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "launches": len(ratios),
    }
    return {
        "kernels": kernels,
        "totals": totals,
        "ops_ratio": ratio if ratios else None,
    }


def table(summary: dict, top: int) -> str:
    """Markdown rows: the `top` GEMMs by time, every attention kernel, the totals."""
    head = "| kind | kernel | launches | ms | share % | TFLOP/s | % peak | clock GHz |"
    out = [head, "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    gemms = [k for k in summary["kernels"] if k["kind"] == "gemm"][:top]
    rest = [k for k in summary["kernels"] if k["kind"] != "gemm"]
    for k in gemms + rest + summary["totals"]:
        out.append(
            f"| {k['kind']} | {k['kernel']} | {k['launches']} | {k['ms']:.2f} | "
            f"{k['share_pct']:.1f} | {k['tflops']:.1f} | {k['pct_peak']:.1f} | "
            f"{k['clock_ghz']:.3f} |"
        )
    return "\n".join(out) + "\n"


def _metric_names(args) -> tuple[str, str]:
    inst = args.inst_metric or INST.get(args.sm)
    ops = args.ops_metric or OPS.get(args.sm)
    if not inst or not ops:
        sys.exit(
            f"no tensor metrics for sm{args.sm}: pass --inst-metric / --ops-metric"
        )
    return inst, ops


def report(args, rep: Path, out: Path) -> dict:
    """Import `rep`, write `tflops.json` and `tflops.md` under `out`; the summary."""
    tool = shlex.split(args.ncu)
    inst, ops = _metric_names(args)
    per_clk = args.flop_per_clk or FLOP_PER_CLK.get(args.sm)
    if not per_clk:
        sys.exit(f"no FLOP/clk/SM for sm{args.sm}: pass --flop-per-clk")
    done = subprocess.run(
        ncu.import_command(tool, rep), capture_output=True, text=True, check=True
    )
    (out / f"{REPORT}.csv").write_text(done.stdout, encoding="utf-8")
    rows = launches(
        done.stdout, args.sm, args.sms, inst, ops, per_clk, args.flop_per_inst
    )
    summ = {
        "sm": args.sm,
        "sms": args.sms,
        "flop_per_clk_per_sm": per_clk,
        "inst_metric": inst,
        "ops_metric": ops,
        **aggregate(rows),
    }
    (out / "tflops.json").write_text(
        json.dumps({**summ, "launches": rows}, indent=1), encoding="utf-8"
    )
    md = table(summ, args.top)
    (out / "tflops.md").write_text(md, encoding="utf-8")
    print(json.dumps(summ["ops_ratio"]))
    print(md)
    return summ


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inst, ops = _metric_names(args)
    child = [
        sys.executable,
        "-u",
        "-m",
        "kohakumoss_bench.run.child",
        "--script",
        args.script,
        "--config",
        args.config,
        "--failure",
        str(out / "failure.json"),
        *(["--env-file", args.env_file] if args.env_file else []),
    ]
    cmd = command(shlex.split(args.ncu), inst, ops, out / REPORT, child, args.regex)
    (out / "cmd.txt").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    with open(out / "log.txt", "w", encoding="utf-8") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        sys.exit(f"ncu exited {code}; see {out / 'log.txt'}")
    report(args, out / f"{REPORT}.ncu-rep", out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "report"))
    ap.add_argument("--sm", type=int, required=True)
    ap.add_argument("--sms", type=int, required=True, help="the card's SM count")
    ap.add_argument("--out", default="", help="output dir (report: the report's dir)")
    ap.add_argument("--report", default="", help="report mode: the .ncu-rep file")
    ap.add_argument(
        "--config", default="", help="run mode: a train config with PROFILE_STEPS"
    )
    ap.add_argument("--script", default="scripts/train/moss.py")
    ap.add_argument("--env-file", default="")
    ap.add_argument("--ncu", default="ncu")
    ap.add_argument("--regex", default=REGEX)
    ap.add_argument("--inst-metric", default="")
    ap.add_argument("--ops-metric", default="")
    ap.add_argument("--flop-per-inst", type=int, default=0)
    ap.add_argument("--flop-per-clk", type=float, default=0.0)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args(argv)
    match args.mode:
        case "run":
            if not args.config or not args.out:
                ap.error("run needs --config and --out")
            run(args)
        case "report":
            if not args.report:
                ap.error("report needs --report")
            rep = Path(args.report)
            report(args, rep, Path(args.out) if args.out else rep.parent)


if __name__ == "__main__":
    main()
