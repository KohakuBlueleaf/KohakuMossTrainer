"""The bench report from the collected host directories
(`runs/bench/<bench>/<host>/`): one table per card pool (speed, VRAM, the step
breakdown, whether the step is compute-, transfer- or software-bound), VRAM targets,
the ncu efficiency pass, quality runs with their grids, inference checks, the kernel
tables, feature coverage, and every failure.

A speed run's result, from its medians and the card's probe:
- transfer share = (h2d GB / pinned h2d GB/s + d2h GB / pinned d2h GB/s) / step time,
  over the DiT engines plus the text encoder and VAE weight traffic;
- compute share = model TFLOP/s / the probe's GEMM TFLOP/s (fp8 for an fp8 base);
- "transfer-bound" when the transfer share >= `transfer_bound`, else "compute-bound"
  when the compute share >= `compute_bound`, else "software-bound".
See docs/performance/bench-matrix.md#report.
"""

import json
from pathlib import Path
from xml.etree import ElementTree

from kohakumoss_bench.measure import ncu
from kohakumoss_bench.plan.config import Options
from kohakumoss_bench.plan.coverage import coverage
from kohakumoss_bench.plan.knobs import script_defaults
from kohakumoss_bench.report.cells import _gib, _num, _phase_vram, _skips
from kohakumoss_bench.report.results import profiled_steps, training_summary
from kohakumoss_bench.report.stage_tables import placement_rows, release_rows

SPEED_KINDS = ("speed", "online", "restart")
MATCH = ("precision", "base", "algo", "profile")
WARMUP = Options().warmup
FP8_BASES = ("fp8_compute", "mxfp8")
FP8_RATE = 2.0


def _read(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def load(root: Path) -> dict:
    """Every host directory under a bench root: cards, probes, kernels, results."""
    hosts = {}
    for host_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        records = []
        for result in sorted(host_dir.glob("*/*/result.json")):
            record = json.loads(result.read_text(encoding="utf-8"))
            record["host"] = host_dir.name
            if (result.parent / "train" / "metrics.jsonl").is_file():
                try:
                    knobs = script_defaults(result.parent / "config.py")
                except (OSError, SyntaxError, ValueError, NameError):
                    knobs = {}
                record.setdefault("summary", {}).update(
                    training_summary(result.parent, WARMUP, profiled_steps(knobs))
                )
            records.append(record)
        review = host_dir / "review-queue.jsonl"
        hosts[host_dir.name] = {
            "cards": _read(host_dir / "cards.json", {}),
            "probes": _read(host_dir / "probes.json", {}),
            "kernels": _read(host_dir / "kernels.json", {}),
            "status": _read(host_dir / "status.json", {}),
            "failure": _read(host_dir / "host-failure.json", None),
            "records": records,
            "review": [
                json.loads(line)
                for line in (
                    review.read_text(encoding="utf-8").splitlines()
                    if review.is_file()
                    else []
                )
                if line.strip()
            ],
        }
    return hosts


def bound_result(
    record: dict, probe: dict, compute_bound: float, transfer_bound: float
) -> dict:
    """Transfer share, compute share and the bound class of one training run."""
    s = record.get("summary", {})
    step = s.get("step_time_median")
    if not step or not probe:
        return {}
    link = probe.get("link", {}).get("pinned", {})
    h2d = sum(
        s.get(k) or 0.0
        for k in ("xfer_h2d_gb_median", "te_h2d_gb_median", "vae_h2d_gb_median")
    )
    d2h = s.get("xfer_d2h_gb_median") or 0.0
    transfer = 0.0
    if link.get("h2d_gbps"):
        transfer += h2d / link["h2d_gbps"]
    if link.get("d2h_gbps"):
        transfer += d2h / link["d2h_gbps"]
    tags = record.get("tags", {})
    gemm = probe.get("gemm_tflops", {})
    peak_key = (
        "fp8"
        if str(tags.get("base", "")).startswith(("fp8", "mxfp8")) and "fp8" in gemm
        else tags.get("precision", "bf16")
    )
    peak = gemm.get(peak_key)
    achieved = s.get("tflops_median")
    compute = achieved / peak if achieved and peak else None
    share = transfer / step
    if share >= transfer_bound:
        cls = "transfer-bound"
    elif compute is not None and compute >= compute_bound:
        cls = "compute-bound"
    else:
        cls = "software-bound"
    return {
        "transfer_share": share,
        "transfer_gb_per_step": h2d + d2h,
        "compute_share": compute,
        "peak_tflops": peak,
        "bound": cls,
    }


def _vram(record: dict) -> tuple:
    """(nvidia-smi peak GiB, steady GiB, torch allocated peak, reserved peak): the
    process column, else (WDDM) the card above its idle baseline."""
    s, smi = record.get("summary", {}), record.get("vram_smi") or {}
    proc = smi.get("smi_process_peak_mib")
    if proc is not None:
        peak, steady = proc, smi.get("smi_process_steady_mib")
    else:
        peak = smi.get("smi_card_above_idle_mib")
        steady = smi.get("smi_card_steady_above_idle_mib")
    return (
        _gib(peak),
        _gib(steady),
        s.get("vram_peak_alloc_max"),
        s.get("vram_peak_reserved_max"),
    )


def speed_rows(records: list[dict], probes: dict, cb: float, tb: float) -> list[str]:
    """The speed table; each run is held against its own card's probe (cards of one
    model can sit on different host links)."""
    head = (
        "| run | card | precision | base | algo | profile | status | samples/s | s/step "
        "| tok/s | MFU (measured MMA) | MFU (nominal) | MFU (fp8 nominal) | of GEMM peak "
        "| smi peak / steady GiB "
        "| alloc / reserved GiB | compute / copy / idle ms | xfer GB/step "
        "| transfer share | bound | prewarm tuned / table | non-finite | loss first -> last |"
    )
    lines = [head, "|" + " --- |" * 23]
    for r in records:
        if r.get("kind") not in SPEED_KINDS:
            continue
        measured = measured_mfu(r, probes)
        t, s = r.get("tags", {}), r.get("summary", {})
        v = bound_result(r, probes.get(r.get("card"), {}), cb, tb)
        peak, steady, alloc, held = _vram(r)
        tr = s.get("trace") or {}
        mfu = s.get("mfu_median")
        fp8 = mfu / FP8_RATE if mfu is not None and t.get("base") in FP8_BASES else None
        r["result"] = v
        lines.append(
            f"| {r['run']} | {r.get('card', '-')} | {t.get('precision', '-')} "
            f"| {t.get('base', '-')} | {t.get('algo', '-')} "
            f"| {t.get('profile', '-')} | {r.get('status')} | {_num(s.get('samples_per_s_median'))} "
            f"| {_num(s.get('step_time_median'), 3)} | {_num(s.get('tokens_per_s_median'), 0)} "
            f"| {_num(measured)} | {_num(mfu)} | {_num(fp8)} | {_num(v.get('compute_share'))} "
            f"| {_num(peak)} / {_num(steady)} | {_num(alloc)} / {_num(held)} "
            f"| {_num(tr.get('trace_compute_ms'), 0)} / {_num(tr.get('trace_copy_ms'), 0)} "
            f"/ {_num(tr.get('trace_idle_ms'), 0)} | {_num(v.get('transfer_gb_per_step'))} "
            f"| {_num(v.get('transfer_share'))} | {v.get('bound', '-')} "
            f"| {s.get('prewarm_tuned', '-')} / {s.get('prewarm_from_table', '-')} "
            f"| {_skips(s)} | {_num(s.get('loss_first'), 4)} -> {_num(s.get('loss_last'), 4)} |"
        )
    return lines


def measured_mfu(record: dict, probes: dict) -> float | None:
    """Model TFLOP/s of a run over its card's measured MMA peak (`mma_tflops` of the
    probe: fp8 for an fp8 base, else the run's precision); None without one."""
    tags = record.get("tags", {})
    peaks = probes.get(record.get("card"), {}).get("mma_tflops", {})
    key = "fp8" if tags.get("base") in FP8_BASES else tags.get("precision")
    tflops = record.get("summary", {}).get("tflops_median")
    peak = peaks.get(key)
    return tflops / peak if tflops and peak else None


def batch_rows(records: list[dict], probes: dict) -> list[str]:
    """Batch scaling: per run its tokens per step and throughput at that budget."""
    lines = [
        "| run | card | MAX_TOKENS | tokens / step | status | samples/s | tok/s "
        "| TFLOP/s | MFU (measured MMA) | MFU (nominal) | smi peak GiB |",
        "|" + " --- |" * 11,
    ]
    for r in records:
        if r.get("kind") != "batch":
            continue
        s = r.get("summary", {})
        tok, step = s.get("tokens_per_s_median"), s.get("step_time_median")
        peak, _, _, _ = _vram(r)
        lines.append(
            f"| {r['run']} | {r.get('card', '-')} | {r.get('tags', {}).get('max_tokens')} "
            f"| {_num(tok * step if tok and step else None, 0)} | {r.get('status')} "
            f"| {_num(s.get('samples_per_s_median'))} | {_num(tok, 0)} "
            f"| {_num(s.get('tflops_median'), 1)} | {_num(measured_mfu(r, probes))} "
            f"| {_num(s.get('mfu_median'))} | {_num(peak)} |"
        )
    return lines


def cache_rows(records: list[dict], kinds: tuple = ("cache",)) -> list[str]:
    """Flagship setups per cache mode: throughput, the text-encoder and VAE time per
    step and share of the wall time (means), the steady step peak (torch), nvidia-smi peak /
    steady of the startup + prewarm phase and of the step loop, the fp16 loss scale."""
    lines = [
        "| run | card | precision | base | algo | profile | cache | status | samples/s "
        "| tok/s | TFLOP/s | te s/step (share) | vae s/step (share) | step peak GiB "
        "| smi prewarm peak / steady GiB | smi step peak / steady GiB | loss scale "
        "| non-finite |",
        "|" + " --- |" * 18,
    ]
    for r in records:
        if r.get("kind") not in kinds:
            continue
        t, s = r.get("tags", {}), r.get("summary", {})
        pre, step = _phase_vram(r, "prewarm"), _phase_vram(r, "step")
        lines.append(
            f"| {r['run']} | {r.get('card', '-')} | {t.get('precision', '-')} "
            f"| {t.get('base', '-')} | {t.get('algo', '-')} | {t.get('profile', '-')} "
            f"| {t.get('cache', '-')} | {r.get('status')} "
            f"| {_num(s.get('samples_per_s_median'))} | {_num(s.get('tokens_per_s_median'), 0)} "
            f"| {_num(s.get('tflops_median'), 1)} "
            f"| {_num(s.get('te_time_mean'), 3)} ({_num(s.get('te_share_mean'))}) "
            f"| {_num(s.get('vae_time_mean'), 3)} ({_num(s.get('vae_share_mean'))}) "
            f"| {_num(s.get('vram_step_peak_median'))} | {_num(pre[0])} / {_num(pre[1])} "
            f"| {_num(step[0])} / {_num(step[1])} | {_num(s.get('loss_scale_median'), 0)} "
            f"| {_skips(s)} |"
        )
    return lines


def ncu_rows(records: list[dict], probes: dict, cb: float, tb: float) -> list[str]:
    """The ncu pass of a pool: totals and result against the matching speed run,
    then its top kernels."""
    lines = []
    speed = [r for r in records if r.get("kind") == "speed"]
    for r in records:
        if r.get("kind") != "ncu":
            continue
        n = r.get("ncu") or {}
        if "top" not in n:
            lines += [f"ncu {r['run']}: {r.get('status')} {n.get('error', '')}", ""]
            continue
        tags = r.get("tags", {})
        twin = next(
            (x for x in speed if all(x["tags"].get(k) == tags.get(k) for k in MATCH)),
            None,
        )
        step = (twin or {}).get("summary", {}).get("step_time_median")
        probe = probes.get((twin or {}).get("card"), {})
        share = (
            bound_result(twin, probe, cb, tb).get("transfer_share") if twin else None
        )
        v = ncu.bound_result(n, step, share, tb)
        c = n["class_share"]
        lines += [
            f"ncu `{n['tensor_metric']}` ({n['chosen_by']}), {r['run']} against "
            f"{(twin or {}).get('run', 'no speed run')}:",
            "",
            "| tensor active (step) % | tensor % (kernel time) | SM % | DRAM % "
            "| kernel ms / step | step s | kernel busy | compute / memory / latency share "
            "| transfer share | result |",
            "|" + " --- |" * 10,
            f"| {_num(v.get('tensor_active_step_pct'), 1)} | {_num(n['tensor_pct'], 1)} "
            f"| {_num(n['sm_pct'], 1)} | {_num(n['dram_pct'], 1)} "
            f"| {_num(n['kernel_ms_per_step'], 0)} | {_num(step, 3)} "
            f"| {_num(v.get('kernel_busy_share'))} | {_num(c['compute'])} / "
            f"{_num(c['memory'])} / {_num(c['latency'])} | {_num(share)} | {v['bound']} |",
            "",
            "| kernel | count | ms / step | time share | tensor % | SM % | DRAM % | class |",
            "|" + " --- |" * 8,
        ]
        lines += [
            f"| `{k['kernel']}` | {k['count']} | {_num(k['ms_per_step'], 1)} "
            f"| {_num(k['time_share'])} | {_num(k['tensor_pct'], 1)} "
            f"| {_num(k['sm_pct'], 1)} | {_num(k['dram_pct'], 1)} | {k['class']} |"
            for k in n["top"]
        ]
        lines.append("")
        r["ncu_result"] = v
    return lines


def cap_rows(records: list[dict]) -> list[str]:
    lines = [
        "| run | target GiB | status | smi GiB | pass | s/step | samples/s |",
        "|" + " --- |" * 7,
    ]
    for r in records:
        if r.get("kind") != "cap":
            continue
        smi, _, _, _ = _vram(r)
        s = r.get("summary", {})
        target = r.get("cap_gib")
        ok = r.get("status") == "ok" and smi is not None and smi <= target
        lines.append(
            f"| {r['run']} | {target} | {r.get('status')} | {_num(smi)} | {'yes' if ok else 'no'} "
            f"| {_num(s.get('step_time_median'), 3)} | {_num(s.get('samples_per_s_median'))} |"
        )
    return lines


def quality_rows(records: list[dict]) -> list[str]:
    lines = [
        "| run | kind | status | steps | non-finite | loss first -> last | output |",
        "|" + " --- |" * 7,
    ]
    for r in records:
        if r.get("kind") not in ("quality", "grid", "infer"):
            continue
        s = r.get("summary", {})
        out = (
            s.get("grid")
            or (s.get("sample_run") or {}).get("generate_s")
            or ", ".join(s.get("exports", [])[:2])
        )
        lines.append(
            f"| {r['run']} | {r['kind']} | {r.get('status')} | {s.get('last_step', '-')} "
            f"| {_skips(s)} | {_num(s.get('loss_first'), 4)} -> {_num(s.get('loss_last'), 4)} | {out or '-'} |"
        )
    return lines


def kernel_rows(kernels: list[dict]) -> list[str]:
    lines = [
        "| kernel | problems | chosen | vs PyTorch median / min |",
        "| --- | --- | --- | --- |",
    ]
    for k in kernels:
        chosen = ", ".join(
            f"{a} x{n}" for a, n in sorted(k["chosen"].items(), key=lambda x: -x[1])
        )
        vs = (
            f"{_num(k.get('vs_pytorch_median'))} / {_num(k.get('vs_pytorch_min'))}"
            if "vs_pytorch_median" in k
            else "no PyTorch path timed"
        )
        lines.append(f"| {k['kernel']} | {k['problems']} | {chosen} | {vs} |")
    return lines


def mma_peaks(probe: dict, peaks: dict[str, dict]) -> dict:
    """The MMA peaks a card's MFU divides by: an externally measured entry of `peaks`
    whose key is part of the card name (first match), else the probe's own MMA-loop
    values that reach at least its cuBLAS GEMM rate (a loop below the GEMM rate did not
    reach the tensor-core peak and is dropped)."""
    name = probe.get("name", "")
    for key, values in peaks.items():
        if key in name:
            return dict(values)
    gemm = probe.get("gemm_tflops", {})
    return {
        k: v for k, v in probe.get("mma_tflops", {}).items() if v >= gemm.get(k, 0.0)
    }


def build(
    root: Path,
    repo: Path,
    junit: dict[str, str],
    cb: float,
    tb: float,
    peaks: dict[str, dict] | None = None,
) -> tuple[str, dict]:
    """(markdown, json) of the whole bench; `peaks` maps a card-name key to its
    measured MMA peaks ({"bf16": TFLOP/s, ...})."""
    hosts = load(root)
    for h in hosts.values():
        for probe in h["probes"].values():
            probe["mma_tflops"] = mma_peaks(probe, peaks or {})
    md = [f"# Bench {root.name}", ""]
    every = []
    for name, h in hosts.items():
        md += [f"## {name}", ""]
        if h["failure"]:
            md += [f"Host runner failed: `{h['failure']}`", ""]
        if h["status"].get("stopped"):
            md += [f"Stopped by an internal failure: `{h['status']['stopped']}`", ""]
        by_model: dict[str, list] = {}
        for r in h["records"]:
            model = r.get("model") or r.get("card", "?").split("-", 1)[-1]
            by_model.setdefault(model, []).append(r)
            every.append(r)
        for model, recs in sorted(by_model.items()):
            slugs = sorted({r.get("card", "?") for r in recs})
            probes = {s: h["probes"][s] for s in slugs if s in h["probes"]}
            md += [f"### {model} (cards {', '.join(slugs)})", ""]
            for slug, p in probes.items():
                link = p.get("link", {}).get("pinned", {})
                gemm = {k: round(v, 1) for k, v in p.get("gemm_tflops", {}).items()}
                md.append(
                    f"- {slug}: GEMM TFLOP/s {json.dumps(gemm)}; pinned link "
                    f"{_num(link.get('h2d_gbps'), 1)} GB/s h2d, "
                    f"{_num(link.get('d2h_gbps'), 1)} GB/s d2h"
                )
            md.append("")
            md += ["Release", "", *release_rows(recs), ""]
            md += ["Cache modes", "", *cache_rows(recs), ""]
            md += ["Encoder placement (no cache)", "", *placement_rows(recs), ""]
            md += ["Speed", "", *speed_rows(recs, probes, cb, tb), ""]
            md += ["Batch scaling", "", *batch_rows(recs, probes), ""]
            md += ["VRAM targets", "", *cap_rows(recs), ""]
            md += ["Efficiency (ncu)", "", *ncu_rows(recs, probes, cb, tb)]
            md += ["Quality, grids, inference", "", *quality_rows(recs), ""]
            if h["kernels"].get(model):
                md += [
                    "Kernels (tuning tables)",
                    "",
                    *kernel_rows(h["kernels"][model]),
                    "",
                ]
        if h["review"]:
            md += ["Review queue", ""]
            md += [
                f"- {e['card']} {e['run']}: {e['status']} {((e.get('failure') or {}).get('message') or '')[:300]}"
                for e in h["review"]
            ]
            md.append("")
    cov = coverage(every, repo, junit)
    md += [
        "## Feature coverage",
        "",
        "| area | feature | by | status | evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in cov:
        md.append(
            f"| {row['area']} | {row['feature']} | {row['by']} | {row['status']} | {'; '.join(row['evidence'][:4])} |"
        )
    data = {
        "hosts": {k: {**v, "records": v["records"]} for k, v in hosts.items()},
        "coverage": cov,
    }
    return "\n".join(md) + "\n", data


def _test_file(case: ElementTree.Element) -> str:
    """The repo-relative test file of a junit testcase (`file`, else its classname)."""
    file = case.get("file")
    if file:
        return file.replace("\\", "/")
    parts = case.get("classname", "").split(".")
    mods = [p for p in parts if p and p[0].islower()]
    return "/".join(mods) + ".py" if mods else ""


def junit_map(path: Path | None) -> dict[str, str]:
    """test file -> "passed" / "failed" from a pytest junit XML (`--junitxml`)."""
    if path is None or not path.is_file():
        return {}
    out: dict[str, str] = {}
    for case in ElementTree.parse(path).getroot().iter("testcase"):
        file = _test_file(case)
        if not file:
            continue
        failed = case.find("failure") is not None or case.find("error") is not None
        if failed:
            out[file] = "failed"
        else:
            out.setdefault(file, "passed")
    return out
