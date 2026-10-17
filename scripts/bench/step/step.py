"""Step bench of the real MOSS architecture (random weights) per memory profile, run
through the trainer (`entry.run`: build, forward, backward and the optimiser step,
synthetic data): step time, tokens/s, peak memory and the memory model's estimate.
Writes runs/bench/step/<gpu>_<variant>.json. See docs/internals/offload.md.

    python scripts/bench/step/step.py --device cuda:0 --profiles balanced,low,minimal
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

from kohakumoss.offload.accel import default_type
from kohakumoss.offload.accel import module as accel
from kohakumoss.training.build.model import workload
from kohakumoss.training.entry import run
from kohakumoss.training.memory.estimate import estimate
from kohakumoss.training.memory.profiles import resolve

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "train" / "moss.py"


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
        WEIGHTS="random",
        VARIANT=args.variant,
        ALGO=args.algo,
        DIM=16,
        ALPHA=16.0,
        PRECISION=args.precision,
        FP16_FIX="online" if args.precision == "fp16" else "none",
        FP16_FIX_KWARGS={"c": 1 / 16} if args.precision == "fp16" else {},
        DATASETS=[{"source": "synthetic", "count": 64}],
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
    return k


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
    return {
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=f"{default_type()}:0")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--algo", default="lora")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--profiles", default="balanced,low,minimal")
    ap.add_argument("--overrides", default="")
    ap.add_argument("--area", type=int, default=1024 * 1024)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--only", default="")
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
    path = os.path.join(
        args.out, f"{report['gpu'].replace(' ', '_')}_{args.variant}_{args.algo}.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
