"""Step bench of MOSS per memory profile, run through the trainer (`entry.run`: build,
forward, backward and the optimiser step) on `--weights` and `--datasets`: step time,
tokens/s, peak memory and the memory model's estimate. Writes
runs/bench/step/<gpu>_<variant>.json. See docs/internals/offload.md.

    python scripts/bench/step/step.py --weights <root>/base \
        --datasets '[{"source": "kohya", "path": "<dataset>"}]' \
        --device cuda:0 --profiles balanced,low,minimal
    python scripts/bench/step/step.py --weights <root>/base --datasets '[...]' \
        --profiles low --text-weights <root>/base --text-plan '{"memory": "ring"}'

With --text-weights the trainer encodes every batch's captions online with the real
text encoder under --text-plan (TEXT_CACHE off): the row then carries the text-encoder
share of the step (`te_ms`, `te_share`, `te_h2d_GB`), the VRAM timeline of the pass
(`vram_te_peak`, `vram_te_after`) and of the DiT step (`vram_step_peak`), and the
estimate with the text encoder.
"""

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel

from kohakumoss.encoders.text import text_plan
from kohakumoss.memory.estimate import estimate
from kohakumoss.memory.planner import with_text
from kohakumoss.memory.profiles import resolve
from kohakumoss.offload.accel import default_type
from kohakumoss.offload.accel import module as accel
from kohakumoss.offload.encoders.text_memory import text_sizes
from kohakumoss.training.build.model import workload
from kohakumoss.training.entry import run

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "train" / "moss.py"
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def script_defaults() -> dict:
    """UPPER_CASE assignments of the train script (its knob block)."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper():
                expr = compile(ast.Expression(node.value), str(SCRIPT), "eval")
                out[name] = eval(expr, {})
    return out


def knobs(args, profile: str, out_dir: str) -> dict:
    """The train script's defaults with the bench's run on top."""
    k = script_defaults()
    k.update(
        WEIGHTS=args.weights,
        VARIANT=args.variant,
        ALGO=args.algo,
        DIM=16,
        ALPHA=16.0,
        PRECISION=args.precision,
        FP16_FIX="online" if args.precision == "fp16" else "none",
        FP16_FIX_KWARGS={"c": 1 / 16} if args.precision == "fp16" else {},
        DATASETS=json.loads(args.datasets),
        LATENT_CACHE="off",
        TEXT_CACHE="off",
        RESOLUTION_MODE="bucket",
        TARGET_AREA=args.area,
        BATCHING="count",
        BATCH_SIZE=1,
        NUM_WORKERS=0,
        STEPS=args.iters + 1,
        LOG_EVERY=1,
        SAVE_EVERY=0,
        SAMPLE_EVERY=0,
        VALIDATE_EVERY=0,
        EXPORT_FORMATS=(),
        LOGGER="console",
        SCHEDULE={"mode": "constant"},
        MEMORY_PROFILE=profile,
        MEMORY_OVERRIDES=json.loads(args.overrides) if args.overrides else {},
        DEVICES=(torch.device(args.device).index or 0,),
        OUTPUT_DIR=out_dir,
        RUN_NAME="step",
    )
    if args.text_weights:
        k.update(TEXT_ENCODER=args.text_weights, TEXT_ENCODE=json.loads(args.text_plan))
    k.update(json.loads(args.knobs) if args.knobs else {})
    return k


def text_estimate(args, k: dict, est, device) -> dict:
    """The memory estimate with the text encoder (sizes from its config on the meta
    device, in the plan's compute dtype)."""
    root = Path(args.text_weights)
    root = root / "text_encoder" if (root / "text_encoder").is_dir() else root
    plan = text_plan(k["TEXT_ENCODE"])
    dtype = DTYPES.get(plan.dtype, torch.bfloat16)
    with torch.device("meta"):
        model = AutoModel.from_config(
            AutoConfig.from_pretrained(str(root)), dtype=dtype
        )
    sizes = text_sizes(getattr(model, "language_model", model), plan.stream_resident)
    tokens = (k["BATCH_SIZE"] * plan.ahead * 300, plan.max_tokens or 16384)
    budget = accel(device).get_device_properties(device).total_memory
    total, memory = with_text(est, sizes, budget, tokens, plan.memory)
    return {"estimate_with_text_GiB": total / 1024**3, "text_memory": memory}


def _mean(rows: list[dict], key: str) -> float:
    """Mean over the logged steps (encode-ahead passes land on some steps only)."""
    vals = [r[key] for r in rows if key in r]
    return sum(vals) / len(vals) if vals else 0.0


def _median(rows: list[dict], key: str) -> float | None:
    vals = sorted(r[key] for r in rows if key in r)
    return vals[len(vals) // 2] if vals else None


def run_profile(args, profile: str) -> dict:
    device = torch.device(args.device)
    with tempfile.TemporaryDirectory() as out:
        k = knobs(args, profile, out)
        run(k)
        lines = (Path(out) / "step" / "metrics.jsonl").read_text().splitlines()
    rows = [json.loads(x) for x in lines][1:]
    med = sorted(r["step_time"] for r in rows)[len(rows) // 2]
    full = args.algo == "full"
    plan = resolve(profile, k["MEMORY_OVERRIDES"], device, trainable_base=full)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.precision)
    est = estimate(workload(k, dtype or torch.float32), plan)
    row = {
        "profile": profile,
        "measures": "trainer step: forward + backward + optimiser step",
        "plan": {key: v for key, v in plan.__dict__.items() if key != "kernels"},
        "ms": med * 1e3,
        "tokens_per_s": sorted(r["tokens_per_s"] for r in rows)[len(rows) // 2],
        "peak_GiB": max((r.get("vram_peak_alloc", 0.0) for r in rows), default=0.0),
        "reserved_GiB": max(
            (r.get("vram_peak_reserved", 0.0) for r in rows), default=0.0
        ),
        "estimate_GiB": est.gib(),
    }
    for key in ("vram_te_peak", "vram_te_after", "vram_step_peak", "vram_step_after"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            row[key + "_GiB"] = max(vals)
    if args.text_weights:
        te = _median(rows, "te_time")
        row["text"] = {
            "plan": k["TEXT_ENCODE"],
            "te_ms": None if te is None else te * 1e3,
            "te_ms_mean": _mean(rows, "te_time") * 1e3,
            "te_share": _median(rows, "te_share"),
            "te_tokens_per_s": _median(rows, "te_tokens_per_s"),
            "te_h2d_GB": _median(rows, "te_h2d_gb"),
            **text_estimate(args, k, est, device),
        }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--datasets", required=True, help="DATASETS (JSON list)")
    ap.add_argument("--device", default=f"{default_type()}:0")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--algo", default="lora")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--profiles", default="balanced,low,minimal")
    ap.add_argument("--overrides", default="")
    ap.add_argument("--knobs", default="", help="JSON of extra train knobs")
    ap.add_argument("--area", type=int, default=1024 * 1024)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--text-weights", default="", help="real text encoder root / dir")
    ap.add_argument("--text-plan", default="{}", help="TextEncoderPlan fields (JSON)")
    ap.add_argument("--out", default="runs/bench/step")
    args, _ = ap.parse_known_args()
    if args.only:
        print("ROW " + json.dumps(run_profile(args, args.only)))
        return
    device = torch.device(args.device)
    rows = []
    for name in args.profiles.split(","):
        cmd = [sys.executable, __file__, "--only", name] + sys.argv[1:]
        out = subprocess.run(cmd, capture_output=True, text=True)
        found = [
            line[4:] for line in out.stdout.splitlines() if line.startswith("ROW ")
        ]
        rows.append(
            json.loads(found[0])
            if found
            else {"profile": name, "error": (out.stdout + out.stderr)[-800:]}
        )
        print(json.dumps(rows[-1]))
    report = {
        "gpu": accel(device).get_device_name(device),
        "args": vars(args),
        "rows": rows,
    }
    os.makedirs(args.out, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = os.path.join(
        args.out,
        f"{report['gpu'].replace(' ', '_')}_{args.variant}_{args.algo}{tag}.json",
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
