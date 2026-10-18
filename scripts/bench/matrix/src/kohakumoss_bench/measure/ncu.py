"""The bench matrix's ncu pass: the Nsight Compute command over one training run's profiler
window, the tensor-pipe metric of the card's chip, and the per-kernel and total
efficiency of the recorded kernels (tensor pipe, SM and DRAM throughput as % of peak,
duration-weighted) with a compute / memory / latency class per kernel.
See docs/performance/bench-matrix.md#ncu.
"""

import csv
import io
import json
import re
import subprocess
from pathlib import Path

DURATION = "gpu__time_duration.sum"
SM = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
PCT = ".avg.pct_of_peak_sustained_elapsed"
TENSOR_NAMES = (
    "sm__pipe_tensor_subpipe_hmma_cycles_active",
    "sm__pipe_tensor_op_hmma_cycles_active",
    "sm__pipe_tensor_cycles_active",
)
CHIPS = {70: "gv100", 75: "tu102", 80: "ga100", 86: "ga102", 89: "ad102", 90: "gh100"}
CHIPS.update({100: "gb100", 120: "gb202"})
REPORT = "ncu"
TOP = 12
# Options.ncu_replay -> the replay flags; see docs/performance/bench-matrix.md#ncu
REPLAYS = {
    "kernel": ("--profile-from-start", "off", "--replay-mode", "kernel"),
    "application": (
        "--profile-from-start",
        "off",
        "--replay-mode",
        "application",
        "--app-replay-mode",
        "relaxed",
    ),
    "app-range": ("--replay-mode", "app-range"),
}


def default_tensor(sm: int) -> str:
    """The tensor-pipe metric name of an arch when the chip query gives none."""
    return TENSOR_NAMES[0] if sm >= 90 else TENSOR_NAMES[1]


def tensor_metric(ncu: list[str], sm: int) -> tuple[str, str]:
    """(metric, how it was chosen) for the card's chip: the first `TENSOR_NAMES` entry
    `ncu --query-metrics --chip <chip>` lists, else `default_tensor`."""
    chip = CHIPS.get(sm)
    if chip is None:
        return default_tensor(sm), f"default (no chip for sm{sm})"
    try:
        done = subprocess.run(
            [*ncu, "--query-metrics", "--chip", chip],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        return default_tensor(sm), f"default (query failed: {err})"
    listed = set(re.findall(r"^\s*(\S+)", done.stdout, flags=re.MULTILINE))
    for name in TENSOR_NAMES:
        if name in listed:
            return name, f"--query-metrics --chip {chip}"
    return (
        default_tensor(sm),
        f"default (not listed for {chip}, exit {done.returncode})",
    )


def metrics(tensor: str) -> list[str]:
    return [DURATION, tensor + PCT, SM, DRAM]


def command(
    ncu: list[str], tensor: str, out: Path, target: list[str], replay: str = "app-range"
) -> list[str]:
    """ncu over `target`'s CUDA profiler window with the `REPLAYS[replay]` flags
    ("kernel" / "application": one result per kernel; "app-range": one result per
    profiler range, the whole run once per pass), clocks and caches left alone, report
    to `<out>.ncu-rep`."""
    if replay not in REPLAYS:
        raise ValueError(f"ncu replay {replay!r}: one of {sorted(REPLAYS)}")
    return [
        *ncu,
        *REPLAYS[replay],
        "--target-processes",
        "all",
        "--clock-control",
        "none",
        "--cache-control",
        "none",
        "--metrics",
        ",".join(metrics(tensor)),
        "--export",
        str(out),
        "--force-overwrite",
        *target,
    ]


def import_command(ncu: list[str], report: Path) -> list[str]:
    """The raw per-kernel CSV of a report (no counters: the binary without sudo)."""
    binary = ncu[-1]
    return [
        binary,
        "--import",
        str(report),
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "base",
    ]


def _number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def parse_csv(text: str) -> list[dict]:
    """Result rows of `ncu --csv --page raw` (one per kernel, or per range under
    app-range): header (the line naming `DURATION`), units row, one row per result
    (lines before the header, such as ==PROF== notes, are skipped)."""
    lines = text.splitlines()
    head_mark = f'"{DURATION}"'
    start = next((i for i, line in enumerate(lines) if head_mark in line), None)
    if start is None:
        return []
    reader = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    if len(reader) < 2:
        return []
    head, units, body = reader[0], reader[1], reader[2:]
    scale = {"usecond": 1e3, "msecond": 1e6, "second": 1e9}
    unit = dict(zip(head, units)).get(DURATION, "nsecond")
    rows = []
    for cells in body:
        row = dict(zip(head, cells))
        dur = _number(row.get(DURATION, ""))
        if dur is None:
            continue
        rows.append(
            {
                "name": row.get("Kernel Name") or row.get("Range Name") or "?",
                "ns": dur * scale.get(unit, 1.0),
                **{k: _number(v) for k, v in row.items() if k.endswith(PCT)},
            }
        )
    return rows


def short(name: str) -> str:
    """A kernel name without its argument list, at most 100 characters."""
    return name.split("(", 1)[0][:100]


def kernel_class(sm: float, dram: float, bound: float) -> str:
    """ "compute" (SM throughput leads and reaches `bound` %), "memory" (DRAM leads and
    reaches it) or "latency" (neither unit near its peak)."""
    if sm >= bound and sm >= dram:
        return "compute"
    if dram >= bound and dram > sm:
        return "memory"
    return "latency"


def summary(rows: list[dict], tensor: str, steps: int, bound: float) -> dict:
    """Totals (duration-weighted tensor / SM / DRAM %, kernel ms per step, the time
    share of each class) and the top kernels by time."""
    t_key = tensor + PCT
    total = sum(r["ns"] for r in rows) or 1.0
    by: dict[str, dict] = {}
    for r in rows:
        sm, dram, ten = (r.get(k) or 0.0 for k in (SM, DRAM, t_key))
        k = by.setdefault(
            short(r["name"]),
            {"count": 0, "ns": 0.0, "tensor": 0.0, "sm": 0.0, "dram": 0.0},
        )
        k["count"] += 1
        k["ns"] += r["ns"]
        k["tensor"] += ten * r["ns"]
        k["sm"] += sm * r["ns"]
        k["dram"] += dram * r["ns"]
    kernels = []
    shares = {"compute": 0.0, "memory": 0.0, "latency": 0.0}
    for name, k in by.items():
        row = {
            "kernel": name,
            "count": k["count"],
            "ms_per_step": k["ns"] / 1e6 / max(steps, 1),
            "time_share": k["ns"] / total,
            "tensor_pct": k["tensor"] / k["ns"] if k["ns"] else 0.0,
            "sm_pct": k["sm"] / k["ns"] if k["ns"] else 0.0,
            "dram_pct": k["dram"] / k["ns"] if k["ns"] else 0.0,
        }
        row["class"] = kernel_class(row["sm_pct"], row["dram_pct"], bound)
        shares[row["class"]] += row["time_share"]
        kernels.append(row)
    kernels.sort(key=lambda r: -r["time_share"])
    return {
        "tensor_metric": tensor,
        "kernels": len(rows),
        "kernels_per_step": len(rows) / max(steps, 1),
        "kernel_ms_per_step": total / 1e6 / max(steps, 1),
        **{
            key: _weighted(kernels, key) for key in ("tensor_pct", "sm_pct", "dram_pct")
        },
        "class_share": shares,
        "top": kernels[:TOP],
        "all": kernels,
    }


def _weighted(kernels: list[dict], key: str) -> float:
    return sum(r[key] * r["time_share"] for r in kernels)


def report_summary(
    ncu: list[str], run_dir: Path, meta: dict, steps: int, bound: float
) -> dict:
    """Import a run's ncu report as CSV (`ncu.csv`) and summarize it (`ncu.json`);
    `meta` plus the summary, or plus an "error"."""
    report = run_dir / f"{REPORT}.ncu-rep"
    if not report.is_file():
        return {**meta, "error": "no ncu report"}
    done = subprocess.run(
        import_command(ncu, report), capture_output=True, text=True, timeout=3600
    )
    (run_dir / "ncu.csv").write_text(done.stdout, encoding="utf-8")
    rows = parse_csv(done.stdout)
    if not rows:
        return {**meta, "error": f"no kernel rows: {done.stderr[-2000:]}"}
    summ = summary(rows, meta["tensor_metric"], steps, bound)
    (run_dir / "ncu.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    return {**meta, **{k: v for k, v in summ.items() if k != "all"}}


def bound_result(
    summ: dict,
    step_s: float | None,
    transfer_share: float | None,
    transfer_bound: float = 0.7,
) -> dict:
    """Whole-step efficiency of the ncu summary against the unprofiled step time:
    tensor active over the step, kernel busy share, and the bound: "PCIe-bound" (the
    speed run's transfer share >= `transfer_bound`), "host-bound" (the kernels fill
    under half of the step), else the class with the largest kernel-time share
    ("compute-bound" / "memory-bound" / "latency-bound")."""
    out = {}
    if step_s:
        busy = summ["kernel_ms_per_step"] / 1e3 / step_s
        out["kernel_busy_share"] = busy
        out["tensor_active_step_pct"] = summ["tensor_pct"] * min(busy, 1.0)
    shares = summ["class_share"]
    lead = max(shares, key=shares.get)
    if transfer_share is not None and transfer_share >= transfer_bound:
        out["bound"] = "PCIe-bound"
    elif out.get("kernel_busy_share", 1.0) < 0.5:
        out["bound"] = "host-bound"
    else:
        out["bound"] = f"{lead}-bound"
    return out
