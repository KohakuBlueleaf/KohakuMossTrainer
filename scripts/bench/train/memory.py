"""Per-phase device memory of the trainer's startup and first steps on real weights:
allocated, peak allocated and reserved GiB after the DiT load, the plan apply, the
trainable set, the text-encoder and VAE loads, the prewarms, every online encode, every
step and the sampling callback, next to the plan's training estimate. One profile per
process (`--profile`); the `--datasets` data, online text and latents.

    python scripts/bench/train/memory.py --weights <root>/base --profile low \
        --datasets '[{"source": "kohya", "path": "<dataset>"}]' \
        --precision bf16 --algo lora --out runs/bench/train
"""

import argparse
import ast
import gc
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from kohakumoss.encoders.text.encoder import QwenTextEncoder
from kohakumoss.memory.estimate import estimate
from kohakumoss.memory.profiles import resolve
from kohakumoss.offload.encoders.vae import PlacedVae
from kohakumoss.training import entry
from kohakumoss.training.build import model as model_build
from kohakumoss.training.callbacks import monitor
from kohakumoss.training.callbacks.evaluate import SampleGrid
from kohakumoss.tuning import prewarm as kernels
from kohakumoss_kernels.hardware.memory import close_window, device_used, open_window

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "train" / "moss.py"
GIB = 2**30
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


def largest_tensors(device: torch.device, n: int = 8) -> list[str]:
    """The `n` largest live tensors on `device` (shape, dtype, GiB), summed per shape."""
    sizes: dict[str, float] = {}
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.device == device:
                key = f"{tuple(obj.shape)} {obj.dtype}"
                sizes[key] = sizes.get(key, 0.0) + obj.numel() * obj.element_size()
        except Exception:  # noqa: BLE001  (symbolic / freed objects)
            continue
    top = sorted(sizes.items(), key=lambda kv: -kv[1])[:n]
    return [f"{k}: {v / GIB:.3f}" for k, v in top]


class Probe:
    """One row per phase, GiB: `allocated` / `reserved` now, `peak` (allocated) and
    `peak_reserved` of the phase's peak window, `device_used` (the whole device, every
    process: nvidia-smi's number). Windows sit on the process-wide peak accumulator, so
    the trainer's own measurements never cut a row."""

    def __init__(self, device: torch.device, dump: tuple[str, ...] = ()):
        self.device = device
        self.dump = dump
        self.rows: list[dict] = []
        self.t0 = time.perf_counter()
        self.window = open_window(device)

    def mark(self, phase: str) -> None:
        torch.cuda.synchronize(self.device)
        alloc, held = close_window(self.window)
        self.window = open_window(self.device)
        self.rows.append(
            {
                "phase": phase,
                "s": round(time.perf_counter() - self.t0, 2),
                "allocated": round(torch.cuda.memory_allocated(self.device) / GIB, 3),
                "peak": round(alloc / GIB, 3),
                "reserved": round(torch.cuda.memory_reserved(self.device) / GIB, 3),
                "peak_reserved": round(held / GIB, 3),
                "device_used": round(device_used(self.device) / GIB, 3),
            }
        )
        print(json.dumps(self.rows[-1]), flush=True)
        if phase in self.dump:
            self.rows[-1]["largest"] = largest_tensors(self.device)
            print(json.dumps(self.rows[-1]["largest"]), flush=True)


def after(owner, name: str, probe: Probe, phase) -> None:
    """`owner.name(...)` marks `phase` (a string, or a function of the call's args)
    after each call."""
    fn = getattr(owner, name)

    def run(*args, **kwargs):
        out = fn(*args, **kwargs)
        probe.mark(phase(*args) if callable(phase) else phase)
        return out

    setattr(owner, name, run)


def instrument(probe: Probe) -> None:
    after(model_build, "build_backbone", probe, "dit_load")
    after(model_build, "apply_to_backbone", probe, "plan_apply")
    after(model_build, "build_trainable", probe, "trainable")
    after(model_build, "place_vae", probe, "vae_load")
    after(model_build, "build_encoders", probe, "encoders")
    load = QwenTextEncoder.from_pretrained.__func__

    def te_load(cls, *args, **kwargs):
        out = load(cls, *args, **kwargs)
        probe.mark("te_load")
        return out

    QwenTextEncoder.from_pretrained = classmethod(te_load)
    after(QwenTextEncoder, "encode", probe, "te_encode")
    after(PlacedVae, "encode_images", probe, "vae_encode")
    after(entry, "build_optim", probe, "optim")
    after(entry, "build_step", probe, "train_step")
    after(entry, "build_ema_for", probe, "ema")
    after(kernels.ModelPrewarm, "__call__", probe, "prewarm")
    after(kernels.AdapterPrewarm, "__call__", probe, "adapter_prewarm")
    after(monitor.Throughput, "on_step_end", probe, lambda s, st, o: f"step_{st.step}")
    after(SampleGrid, "on_step_end", probe, lambda s, st, o: f"sample_{st.step}")


def knobs(args, out_dir: str) -> dict:
    k = script_defaults()
    k.update(
        WEIGHTS=args.weights,
        VARIANT="base",
        ALGO=args.algo,
        DIM=16,
        ALPHA=16.0,
        PRECISION=args.precision,
        DATASETS=json.loads(args.datasets),
        LATENT_CACHE="off",
        TEXT_CACHE="off",
        RESOLUTION_MODE="bucket",
        TARGET_AREA=args.area,
        BATCHING="count",
        BATCH_SIZE=1,
        NUM_WORKERS=0,
        STEPS=args.steps,
        LOG_EVERY=1,
        SAVE_EVERY=0,
        SAMPLE_EVERY=args.sample_every,
        SAMPLE_STEPS=2,
        SAMPLE_PROMPTS=["a watercolor painting of a small cottage beside a lake"],
        SAMPLE_SIZE=(1024, 1024),
        VALIDATE_EVERY=0,
        EXPORT_FORMATS=(),
        LOGGER="console",
        SCHEDULE={"mode": "constant"},
        MEMORY_PROFILE=args.profile,
        DEVICES=(0,),
        OUTPUT_DIR=out_dir,
        RUN_NAME="memory",
    )
    k.update(json.loads(args.knobs) if args.knobs else {})
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--datasets", required=True, help="DATASETS (JSON list)")
    ap.add_argument("--profile", default="balanced")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--algo", default="lora")
    ap.add_argument("--area", type=int, default=1024 * 1024)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--sample-every", type=int, default=3)
    ap.add_argument("--knobs", default="", help="JSON of extra train knobs")
    ap.add_argument("--dump", default="", help="phases that list the largest tensors")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="runs/bench/train")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    probe = Probe(dev, tuple(args.dump.split(",")) if args.dump else ())
    probe.mark("start")
    instrument(probe)
    with tempfile.TemporaryDirectory() as out:
        k = knobs(args, out)
        entry.run(k)
    plan = resolve(
        args.profile, k.get("MEMORY_OVERRIDES"), dev, trainable_base=args.algo == "full"
    )
    work = model_build.workload(k, DTYPES.get(args.precision, torch.bfloat16))
    est = estimate(work, plan).gib()
    steps = [r for r in probe.rows if r["phase"].startswith("step_")]
    report = {
        "args": vars(args),
        "gpu": torch.cuda.get_device_name(dev),
        "rows": probe.rows,
        "step_peak_GiB": max((r["peak"] for r in steps), default=None),
        "max_peak_GiB": max(r["peak"] for r in probe.rows),
        "max_reserved_GiB": max(r["peak_reserved"] for r in probe.rows),
        "max_device_used_GiB": max(r["device_used"] for r in probe.rows),
        "estimate_GiB": est,
    }
    os.makedirs(args.out, exist_ok=True)
    name = f"memory_{args.profile}_{args.algo}_{args.precision}{args.tag}.json"
    with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
