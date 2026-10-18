"""Scaling table of a set of training runs: every run directory under ROOT (holding
config.json, metrics.jsonl and ranks/summary.json) becomes one row per engine x GPU
count x precision x algorithm x memory profile, with samples/s, scaling efficiency
against the single-GPU run of the same settings and memory profile, the step
breakdown, peak VRAM and the loss parity
of the first PARITY_STEPS steps against the single-GPU run that sees the same global
batch (GRAD_ACC = data ranks x GRAD_ACC, or MICROBATCHES for the pipelines).

    kogine run scripts/dist/scaling_report.py --set ROOT=runs/scale --set HOST=v100
Writes <ROOT>/scaling.json and <ROOT>/scaling.md.
"""

import json
import math
from pathlib import Path

from kohakumoss.logging.log import get_logger, setup
from kohakumoss_common.knobs import check_script

log = get_logger("scripts.scaling_report")

ROOT = "runs/scale"  # directory whose subdirectories are run outputs
HOST = ""  # label of the host the runs ran on
PARITY_STEPS = 20  # leading steps compared against the parity reference
MATCH = ("PRECISION", "ALGO", "MAX_TOKENS", "TARGET_AREA")
BREAKDOWN = (
    "ddp_comm_wait_ms",
    "ddp_span_ms",
    "pipe_compute_ms",
    "pipe_idle_ms",
    "pipe_bubble_frac",
    "trace_compute_ms",
    "trace_comm_ms",
    "trace_comm_exposed_ms",
    "trace_idle_ms",
    "trace_compute_frac",
)


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def load(run: Path) -> dict | None:
    """One run's config, rank-0 metric rows, summary and per-rank breakdown means."""
    cfg_path, summary_path = run / "config.json", run / "ranks" / "summary.json"
    if not (cfg_path.exists() and summary_path.exists()):
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trace = {}
    for f in sorted((run / "ranks").glob("trace.rank*.breakdown.json")):
        rank = f.name.split(".")[1]
        trace[rank] = json.loads(f.read_text(encoding="utf-8"))
    return {
        "name": run.name,
        "cfg": cfg,
        "rows": _rows(run / "metrics.jsonl"),
        "summary": summary,
        "trace": trace,
    }


def gpus(cfg: dict) -> int:
    return len(cfg["DEVICES"])


def global_acc(cfg: dict) -> int:
    """Micro-batches of one optimiser step over the whole job (the single-GPU
    GRAD_ACC that sees the same global batch)."""
    match cfg["PARALLEL"]:
        case "pipeline" | "threaded":
            return int(cfg["MICROBATCHES"])
        case "ddp" | "fsdp":
            return gpus(cfg) * int(cfg["GRAD_ACC"])
    return int(cfg["GRAD_ACC"])


def singles(runs: list[dict], ref: dict, match=MATCH) -> list[dict]:
    """The single-GPU runs with `ref`'s settings (`match` knobs), by GRAD_ACC."""
    found = [
        r
        for r in runs
        if r["cfg"]["PARALLEL"] == "none"
        and all(r["cfg"].get(k) == ref["cfg"].get(k) for k in match)
    ]
    return sorted(found, key=lambda r: int(r["cfg"]["GRAD_ACC"]))


def single(runs: list[dict], ref: dict, acc: int) -> dict | None:
    """The single-GPU run with `ref`'s settings and GRAD_ACC = `acc`."""
    return next(
        (r for r in singles(runs, ref) if int(r["cfg"]["GRAD_ACC"]) == acc), None
    )


def baseline(runs: list[dict], ref: dict) -> dict | None:
    """The single-GPU speed reference on the same memory profile: GRAD_ACC 1, else the
    smallest GRAD_ACC."""
    found = singles(runs, ref, (*MATCH, "MEMORY_PROFILE"))
    return found[0] if found else None


def parity(run: dict, ref: dict | None, steps: int) -> dict:
    """Relative loss difference per step over the first `steps` steps (max, mean) and
    the two runs' mean losses."""
    if ref is None:
        return {}
    a = {r["step"]: r["loss"] for r in run["rows"] if r["step"] <= steps}
    b = {r["step"]: r["loss"] for r in ref["rows"] if r["step"] <= steps}
    common = sorted(set(a) & set(b))
    rel = [abs(a[s] - b[s]) / max(abs(b[s]), 1e-12) for s in common]
    rel = [v for v in rel if math.isfinite(v)]
    if not rel:
        return {"parity_ref": ref["name"], "parity_steps": 0}
    return {
        "parity_ref": ref["name"],
        "parity_steps": len(rel),
        "loss_rel_max": max(rel),
        "loss_rel_mean": sum(rel) / len(rel),
        "loss_mean": sum(a[s] for s in common) / len(common),
        "ref_loss_mean": sum(b[s] for s in common) / len(common),
    }


def row(run: dict, runs: list[dict]) -> dict:
    cfg, job = run["cfg"], run["summary"]["job"]
    ranks = run["summary"]["ranks"]
    base = baseline(runs, run)
    piped = cfg["PARALLEL"] in ("pipeline", "threaded")
    out = {
        "host": HOST,
        "run": run["name"],
        "engine": cfg["PARALLEL"],
        "schedule": cfg["PIPELINE_SCHEDULE"] if piped else "",
        "profile": cfg["MEMORY_PROFILE"],
        "gpus": gpus(cfg),
        "precision": cfg["PRECISION"],
        "algo": cfg["ALGO"],
        "grad_acc": global_acc(cfg),
        "samples_per_s": job["samples_per_s"],
        "tokens_per_s": job["tokens_per_s"],
        "step_time": job["step_time"],
        "mfu": job.get("mfu"),
        "smi_mem_peak_gib": job.get("smi_mem_peak_gib"),
        "vram_peak_alloc_gib": job.get("vram_peak_alloc_gib"),
        "vram_peak_reserved_gib": job.get("vram_peak_reserved_gib"),
        "host_ram_gib": job.get("host_ram_gib"),
    }
    private = [r["host_private_gib"] for r in ranks if "host_private_gib" in r]
    if private:
        out["host_private_gib_rank"] = max(private)
    if base is not None and base is not run:
        one = base["summary"]["job"]["samples_per_s"]
        out["baseline"] = base["name"]
        out["scaling_eff"] = job["samples_per_s"] / max(out["gpus"] * one, 1e-12)
        out["speedup"] = job["samples_per_s"] / max(one, 1e-12)
    for key in BREAKDOWN:
        vals = [r[key] for r in ranks if key in r]
        vals += [t[key] for t in run["trace"].values() if key in t]
        if vals:
            out[key] = max(vals)
    if cfg["PARALLEL"] != "none" or int(cfg["GRAD_ACC"]) > 1:
        ref = single(runs, run, global_acc(cfg))
        out.update(parity(run, ref if ref is not run else None, PARITY_STEPS))
    return out


def markdown(rows: list[dict]) -> str:
    cols = (
        "engine",
        "schedule",
        "profile",
        "gpus",
        "precision",
        "algo",
        "samples_per_s",
        "scaling_eff",
        "step_time",
        "mfu",
        "ddp_comm_wait_ms",
        "trace_compute_ms",
        "trace_comm_exposed_ms",
        "pipe_bubble_frac",
        "smi_mem_peak_gib",
        "host_private_gib_rank",
        "host_ram_gib",
        "loss_rel_max",
    )
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    setup()
    check_script(globals())
    root = Path(ROOT)
    runs = [r for r in (load(p) for p in sorted(root.iterdir()) if p.is_dir()) if r]
    rows = sorted(
        (row(r, runs) for r in runs),
        key=lambda r: (
            r["algo"],
            r["precision"],
            r["engine"],
            r["profile"],
            r["gpus"],
            r["run"],
        ),
    )
    (root / "scaling.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    table = markdown(rows)
    (root / "scaling.md").write_text(table, encoding="utf-8")
    log.info("\n" + table)


if __name__ == "__main__":
    main()
