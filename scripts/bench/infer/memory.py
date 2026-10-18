"""Per-phase device memory of one MossPipeline generate call: allocated, peak allocated
and reserved GiB after load, text encode, every sampler step and the VAE decode, plus
whether any tensor or autograd state was produced.

    python scripts/bench/infer/memory.py --weights <root>/base --transformer <dit file> \
        --dtype fp16 --memory balanced --vae-memory offload --text-memory stream \
        --size 1024x1024 --steps 4 --out runs/bench/infer
    python scripts/bench/infer/memory.py --weights <root>/base --memory minimal \
        --precision int8 --out runs/bench/infer

`--precision` sets the DiT base (as PRECISION in scripts/sample/sample.py: a quantized
base without `--memory` runs under `speed`).
"""

import argparse
import json
import os
import time

import torch

import kohakumoss.inference.loading as L
from kohakumoss.encoders import QwenTextEncoder
from kohakumoss.inference import MossPipeline
from kohakumoss.memory.estimate import Workload, estimate_inference
from kohakumoss.memory.profiles import resolve
from kohakumoss_kernels.hardware.memory import close_window, device_used, open_window

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
GIB = 2**30


class Probe:
    """One row per phase on `device`, GiB: `allocated` / `reserved` now, `peak`
    (allocated) and `peak_reserved` of the phase's peak window (on the process-wide
    accumulator, so the pipeline's own measurement never cuts a row), `device_used`
    (the whole device, every process: nvidia-smi's number)."""

    def __init__(self, device: torch.device):
        self.device = device
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
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        print(json.dumps(self.rows[-1]), flush=True)


def wrap(obj, name: str, probe: Probe, phase: str) -> None:
    """`obj.name(...)` marks `phase` after each call."""
    fn = getattr(obj, name)

    def run(*args, **kwargs):
        out = fn(*args, **kwargs)
        probe.mark(phase)
        return out

    setattr(obj, name, run)


_PROBE: list[Probe] = []
_TE_LOAD = QwenTextEncoder.from_pretrained.__func__


def _probed_te(cls, *args, **kwargs):
    out = _TE_LOAD(QwenTextEncoder, *args, **kwargs)
    _PROBE[0].mark("text_encoder_load")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--transformer", default="")
    ap.add_argument("--dtype", default="fp16", choices=tuple(DTYPES))
    ap.add_argument("--memory", default="")
    ap.add_argument(
        "--precision",
        default="compute",
        choices=("compute", "int8", "fp8", "fp8_compute", "mxfp8"),
    )
    ap.add_argument("--vae-memory", default="offload")
    ap.add_argument("--vae-tile", action="store_true")
    ap.add_argument("--text-memory", default="ring")
    ap.add_argument("--text-precision", default="compute")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--tune", default="off", choices=("off", "table"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="runs/bench/infer")
    args = ap.parse_args()
    quantized = args.precision != "compute"
    memory = args.memory or ("speed" if quantized else None)
    overrides = {"base_precision": args.precision} if quantized else {}
    dev = torch.device("cuda:0")
    probe = Probe(dev)
    probe.mark("start")
    wrap(L, "load_backbone", probe, "load_backbone")
    wrap(L, "load_transformer", probe, "load_transformer")
    wrap(L, "apply_inference_plan", probe, "apply_inference_plan")
    L.QwenTextEncoder = type(
        "Probed", (L.QwenTextEncoder,), {"from_pretrained": classmethod(_probed_te)}
    )
    _PROBE.append(probe)
    wrap(L.FluxVae, "from_pretrained", probe, "vae_load")
    pipe = MossPipeline.from_pretrained(
        args.weights,
        transformer=args.transformer or None,
        dtype=DTYPES[args.dtype],
        device=dev,
        vae_plan={"memory": args.vae_memory, "tile": args.vae_tile},
        memory=memory,
        memory_overrides=overrides,
        text_plan={"memory": args.text_memory, "precision": args.text_precision},
        tune=args.tune,
    )
    probe.mark("load")
    wrap(pipe.place, "encode", probe, "text_encode")
    wrap(pipe.place, "decode", probe, "vae_decode")
    h, w = (int(v) for v in args.size.split("x"))
    grads: list[str] = []

    def on_step(i, x):
        if x.requires_grad or x.grad_fn is not None:
            grads.append(f"step {i}")
        probe.mark(f"step_{i}")

    result = pipe.generate(
        ["a watercolor painting of a small cottage beside a lake at sunrise"],
        negative_prompts=["blurry, distortion"],
        sizes=[(h, w)],
        steps=args.steps,
        seeds=[0],
        batch=1,
        step_callback=on_step,
    )
    probe.mark("done")
    leaked = [n for n, p in pipe.backbone.named_parameters() if p.requires_grad]
    estimate = None
    if memory:
        plan = resolve(memory, overrides, dev, inference=True)
        work = Workload(
            pipe.backbone.cfg, img_tokens=2 * (h // 16) * (w // 16), txt_tokens=64
        )
        estimate = estimate_inference(work, plan).gib()
    report = {
        "args": vars(args),
        "gpu": torch.cuda.get_device_name(dev),
        "rows": probe.rows,
        "max_peak_GiB": max(r["peak"] for r in probe.rows),
        "max_reserved_GiB": max(r["peak_reserved"] for r in probe.rows),
        "max_device_used_GiB": max(r["device_used"] for r in probe.rows),
        "generate_vram": result.settings.get("vram"),
        "dit_estimate_GiB": estimate,
        "params_requiring_grad": len(leaked),
        "steps_with_autograd": grads,
        "latent_requires_grad": bool(result.latents[0].requires_grad),
        "step_stats": result.settings.get("step_stats"),
    }
    os.makedirs(args.out, exist_ok=True)
    name = f"memory_{memory or 'none'}_{args.precision}_{args.dtype}{args.tag}.json"
    with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
