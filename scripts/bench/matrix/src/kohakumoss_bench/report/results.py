"""Summaries of finished bench runs, from what the run wrote: the training metric
rows (`<run>/train/metrics.jsonl`), the step-1 breakdown, the profiled window's kernel
timeline (`trace_*`), the per-rank summary, the run log (kernel and adapter prewarm
lines), the sampler trace and `metadata*.json` of the sample scripts.
See docs/performance/bench-matrix.md#results.
"""

import json
import math
import re
import statistics
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PREWARM = re.compile(
    r"prewarm: (\d+) problems from the tuning table, (\d+) tuned in ([0-9.]+) s"
)
ADAPTER_OPS = re.compile(
    r"adapter kernel ops: (\d+) autotuned, (\d+) distinct, (\d+) problems measured"
)
COUNTERS = ("step", "skipped_steps", "nonfinite_steps", "nonfinite_params")
MEAN_KEYS = ("te_time", "te_share", "te_h2d_gb", "vae_time", "vae_share", "vae_h2d_gb")
PLACEMENT = re.compile(r"(text encoder|VAE) placement: (\w+)")
STEP_ONE = re.compile(
    r"VRAM peak: prewarm ([0-9.]+) GiB allocated / ([0-9.]+) reserved; "
    r"step 1 ([0-9.]+) / ([0-9.]+)"
)
MIB_GIB = 1024.0
MAX_KEYS = (
    "vram_peak_alloc",
    "vram_peak_reserved",
    "vram_device",
    "vram_non_torch",
    "vram_fragmented",
    "vram_cached",
    "prewarm_peak_alloc",
    "prewarm_peak_reserved",
)


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _finite(rows: list[dict], key: str) -> list[float]:
    return [
        r[key]
        for r in rows
        if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])
    ]


def _median(rows: list[dict], key: str) -> float | None:
    vals = _finite(rows, key)
    return statistics.median(vals) if vals else None


def _mean(rows: list[dict], key: str) -> float | None:
    vals = _finite(rows, key)
    return statistics.fmean(vals) if vals else None


def _max(rows: list[dict], key: str) -> float | None:
    vals = _finite(rows, key)
    return max(vals) if vals else None


def _top(*values) -> float | None:
    known = [v for v in values if v is not None]
    return max(known) if known else None


def profiled_steps(knobs: dict) -> set[int]:
    """Steps inside the run's TRACE_STEPS / PROFILE_STEPS windows (slower: profiled)
    and the step after each window (it holds the trace export)."""
    out: set[int] = set()
    for name in ("TRACE_STEPS", "PROFILE_STEPS"):
        window = tuple(knobs.get(name) or ())
        if window:
            out.update(range(int(window[0]) + 1, int(window[1]) + 2))
    return out


def _metric_keys(rows: list[dict]) -> list[str]:
    keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float))}
    return sorted(k for k in keys - set(COUNTERS) if not k.startswith("trace_"))


def nonfinite_at(rows: list[dict]) -> list[tuple]:
    """(step, loss scale after it) of every metric row with a non-finite grad norm."""
    return [
        (r["step"], r.get("loss_scale"))
        for r in rows
        if isinstance(r.get("grad_norm"), float) and not math.isfinite(r["grad_norm"])
    ]


def training_summary(
    run_dir: Path, warmup: int, profiled: frozenset | set = frozenset()
) -> dict:
    """Medians of every numeric metric over the steps after `warmup` outside the
    `profiled` steps, means of the encoder times and shares there (`MEAN_KEYS`: an
    encoder running `ahead` batches at once costs 0 on most steps), maxima over the
    run, counts, loss ends, the profiled window's kernel timeline, the step-1 breakdown
    and the per-rank summary."""
    root = run_dir / "train"
    rows = read_rows(root / "metrics.jsonl")
    if not rows:
        return {"rows": 0}
    steady = [
        r for r in rows if r.get("step", 0) > warmup and r.get("step") not in profiled
    ] or rows
    out = {"rows": len(rows), "last_step": rows[-1].get("step")}
    out["steady_steps"] = len(steady)
    out.update({f"{k}_median": _median(steady, k) for k in _metric_keys(steady)})
    out.update({f"{k}_mean": _mean(steady, k) for k in MEAN_KEYS})
    out.update({f"{k}_max": _max(rows, k) for k in MAX_KEYS})
    later = [r for r in rows if r.get("step", 0) >= 2]
    out["steps_alloc_max"] = _top(
        _max(later, "vram_step_peak"), _max(later, "vram_peak_alloc")
    )
    out["steps_reserved_max"] = _max(later, "vram_peak_reserved")
    reserved = _finite(later, "vram_peak_reserved")
    out["steps_reserved_first"] = reserved[0] if reserved else None
    out["steps_reserved_last"] = reserved[-1] if reserved else None
    traced = [r for r in rows if "trace_wall_ms" in r]
    if traced:
        out["trace"] = {k: v for k, v in traced[-1].items() if k.startswith("trace_")}
    out["step1_time"] = rows[0].get("step_time")
    last = rows[-1]
    out["nonfinite_steps"] = last.get("nonfinite_steps")
    out["skipped_steps"] = last.get("skipped_steps")
    finite = _finite(rows, "loss")
    out["loss_first"] = finite[0] if finite else None
    out["loss_last"] = finite[-1] if finite else None
    out["loss_nan_rows"] = len(rows) - len(finite)
    out["nonfinite_at"] = nonfinite_at(rows)
    breakdown = root / "startup" / "step1_breakdown.json"
    if breakdown.is_file():
        out["step1_breakdown"] = json.loads(breakdown.read_text(encoding="utf-8"))
    ranks = root / "ranks" / "summary.json"
    if ranks.is_file():
        out["rank_summary"] = json.loads(ranks.read_text(encoding="utf-8"))
    return out


def loop_start(run_dir: Path) -> float | None:
    """Unix time the training loop's first step began (its metric row's `time` less its
    `step_time`); None without a timed first row."""
    rows = read_rows(run_dir / "train" / "metrics.jsonl")
    if not rows or not isinstance(rows[0].get("time"), (int, float)):
        return None
    return rows[0]["time"] - (rows[0].get("step_time") or 0.0)


def log_summary(log_path: Path) -> dict:
    """The kernel and adapter prewarm lines and the text-encoder / VAE placement lines
    of a run log."""
    if not log_path.is_file():
        return {}
    text = ANSI.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    out = {}
    found = PREWARM.findall(text)
    if found:
        table, tuned, secs = found[-1]
        out.update(
            prewarm_from_table=int(table),
            prewarm_tuned=int(tuned),
            prewarm_s=float(secs),
        )
    for which, memory in PLACEMENT.findall(text):
        out["te_placement" if which == "text encoder" else "vae_placement"] = memory
    step_one = STEP_ONE.findall(text)
    if step_one:
        _, _, alloc, held = step_one[-1]
        out.update(step1_peak_alloc=float(alloc), step1_peak_reserved=float(held))
    adapter = ADAPTER_OPS.findall(text)
    if adapter:
        ops, distinct, measured = adapter[-1]
        out.update(
            adapter_ops=int(ops),
            adapter_distinct=int(distinct),
            adapter_measured=int(measured),
        )
    return out


def _smi_peak(phases: dict, phase: str) -> float | None:
    mib = phases.get(f"{phase}_smi_process_peak_mib")
    if mib is None:
        mib = phases.get(f"{phase}_smi_card_above_idle_mib")
    return None if mib is None else mib / MIB_GIB


def phase_peaks(record: dict) -> dict:
    """GiB of a training run's prewarm phase against its step phase: the nvidia-smi
    peak (the process, else the card above idle), torch max allocated / reserved (the
    prewarm's; the steps': step 1's own peak and every later step's), and `over`: True
    when a prewarm peak exceeds the step one, None without a comparable pair (the rule:
    tuning never uses more VRAM than training); torch reserved at the first and the last
    step after step 1 (the growth trend), and `past_budget`: for a run under a VRAM cap,
    whether the step phase's smi peak or reserved peak went past it.
    See docs/performance/bench-matrix.md#release."""
    s, phases = record.get("summary", {}), record.get("vram_smi_phases") or {}
    out = {
        "smi_prewarm": _smi_peak(phases, "prewarm"),
        "smi_step": _smi_peak(phases, "step"),
        "torch_prewarm_alloc": s.get("prewarm_peak_alloc_max"),
        "torch_prewarm_reserved": s.get("prewarm_peak_reserved_max"),
        "torch_step_alloc": _top(s.get("step1_peak_alloc"), s.get("steps_alloc_max")),
        "torch_step_reserved": _top(
            s.get("step1_peak_reserved"), s.get("steps_reserved_max")
        ),
    }
    pairs = [
        (out[f"{k}_prewarm{x}"], out[f"{k}_step{x}"])
        for k, x in (("smi", ""), ("torch", "_alloc"), ("torch", "_reserved"))
    ]
    known = [(a, b) for a, b in pairs if a is not None and b is not None]
    out["over"] = any(a > b for a, b in known) if known else None
    out["reserved_first"] = s.get("steps_reserved_first")
    out["reserved_last"] = s.get("steps_reserved_last")
    cap = record.get("cap_gib") or 0.0
    seen = [v for v in (out["smi_step"], out["torch_step_reserved"]) if v is not None]
    out["past_budget"] = (max(seen) > cap) if cap and seen else None
    return out


def sample_summary(run_dir: Path) -> dict:
    """Generate time / VRAM records of a sample.py or compare_grid run."""
    out = {}
    metas = sorted(run_dir.glob("images/metadata*.json"))
    if metas:
        meta = json.loads(metas[-1].read_text(encoding="utf-8"))
        out["sample_run"] = meta.get("run", meta)
    grid = run_dir / "grid" / "grid.png"
    if grid.is_file():
        out["grid"] = str(grid)
    return out


def exports_of(run_dir: Path) -> list[str]:
    """Exported adapter files of a training run."""
    root = run_dir / "train" / "export"
    return sorted(str(p) for p in root.glob("*.safetensors")) if root.is_dir() else []
