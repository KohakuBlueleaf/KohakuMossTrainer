"""Text-encoder quantization study: per precision mode (int8 / fp8 weight-only, per
channel or per group, fp8 compute) the embedding error against the bf16 encode and
against an fp64 probe reference, encode speed, VRAM and stored bytes, and the effect
on the DiT: velocity error on fixed latents and LPIPS of samples against the bf16-TE
samples (same seeds).

    python scripts/bench/textenc/quant.py --root <weights>/base \
        --precisions compute,int8,fp8,int8:128,fp8:128,fp8_compute \
        --memory ring --out runs/bench/textenc

`--ref64` takes a saved list of fp64 embeddings of `--probe` prompts (one tensor per
prompt, e.g. from the fp64 kernel forward on the host); without it the fp64 columns
are skipped. Writes <out>/<gpu>_quant.json.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import lpips
import torch

from kohakumoss.encoders.text import QwenTextEncoder, text_plan
from kohakumoss.inference import MossPipeline
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels.hardware.memory import device_used, window

acc = torch.accelerator

PROMPTS = [
    "1girl, solo, long hair, looking at viewer, smile, outdoors, cherry blossoms",
    "a red fox sitting in fresh snow at dawn, telephoto photo, soft light",
    "an old lighthouse on a cliff during a storm, oil painting, dramatic clouds",
    "a bowl of ramen with a soft egg, overhead shot, studio lighting",
]


def parse(mode: str) -> dict:
    precision, _, group = mode.partition(":")
    return {"precision": precision, "quant_group": int(group or 0)}


def rel(a, b) -> tuple[float, float]:
    """(relative RMS error, 1 - cosine) of a against b over all tokens."""
    a, b = a.double().flatten(), b.double().flatten()
    rms = ((a - b).norm() / b.norm()).item()
    cos = (a @ b / (a.norm() * b.norm())).item()
    return rms, 1.0 - cos


def encode(args, mode, prompts, device):
    gc.collect()
    acc.empty_cache()
    spec = {"memory": args.memory, "dtype": args.dtype, **parse(mode)}
    te = QwenTextEncoder.from_pretrained(
        Path(args.root) / "text_encoder", device, text_plan(spec), output_device="cpu"
    )
    te.encode(prompts)
    base = acc.memory_allocated(device)
    t0 = time.perf_counter()
    with window(device) as w:
        out = te.encode(prompts)
    dt = time.perf_counter() - t0
    row = {
        "mode": mode,
        "memory": te.plan.memory,
        "s_per_pass": dt,
        "tokens_per_s": sum(o.shape[0] for o in out) / dt,
        "peak_GiB": (w.alloc - base) / 2**30,
        "peak_reserved_GiB": w.reserved / 2**30,
        "device_used_GiB": device_used(device) / 2**30,
        "pass_weights_GiB": te.footprint()["pass_bytes"] / 2**30,
        "quant": te.quant,
    }
    empty = te.empty()
    te.release()
    return out, empty, row


def velocity(pipe, embeds, device, seed=0):
    """DiT velocity on fixed latents / sigma for every prompt embedding."""
    cfg = pipe.backbone.cfg
    dtype = next(pipe.backbone.parameters()).dtype
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for emb in embeds:
        seq = build_seqinfo(torch.tensor([[32, 32]]), torch.tensor([emb.shape[0]]))
        seq = seq.to(device)
        z = torch.randn(seq.n_img, cfg.latent_channels, generator=g).to(device)
        sigma = torch.tensor([0.6], device=device)
        with torch.no_grad():
            v = pipe.backbone(z.to(dtype), emb.to(device, dtype), sigma, seq)
        out.append(v.float().cpu())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="pipeline dir (text_encoder/ ...)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precisions", default="compute,int8,fp8,int8:128,fp8:128")
    ap.add_argument("--memory", default="ring")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--ref64", default="", help="torch.save'd fp64 embeddings")
    ap.add_argument("--probe", default="", help="JSON list of the --ref64 prompts")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="runs/bench/textenc")
    args = ap.parse_args()
    device = torch.device(args.device)
    probe = json.loads(Path(args.probe).read_text()) if args.probe else []
    ref64 = torch.load(args.ref64) if args.ref64 else None
    modes = args.precisions.split(",")
    embeds, empties, rows = {}, {}, []
    for mode in modes:
        out, empties[mode], row = encode(args, mode, PROMPTS + probe, device)
        embeds[mode] = out[: len(PROMPTS)]
        if ref64 is not None:
            got = torch.cat(out[len(PROMPTS) :])
            row["fp64_rel_rms"], row["fp64_cos_dist"] = rel(got, torch.cat(ref64))
        rows.append(row)
        print(json.dumps(row), flush=True)
    base = modes[0]
    pipe = MossPipeline.from_pretrained(
        args.root, device=device, text_encoder=False, memory=None
    )
    metric = lpips.LPIPS(net="alex").to(device)
    v_ref = velocity(pipe, embeds[base], device)
    images = {}
    for mode in modes:
        res = pipe.generate(
            prompt_embeds=[e.to(device) for e in embeds[mode]],
            negative_embeds=[empties[mode].to(device)] * len(PROMPTS),
            sizes=(args.size, args.size),
            steps=args.steps,
            seeds=list(range(len(PROMPTS))),
            output="tensor",
        )
        images[mode] = res.images
    for row in rows:
        mode = row["mode"]
        a, b = torch.cat(embeds[mode]), torch.cat(embeds[base])
        row["bf16_rel_rms"], row["bf16_cos_dist"] = rel(a, b)
        v = velocity(pipe, embeds[mode], device)
        row["dit_v_rel_rms"] = rel(torch.cat(v), torch.cat(v_ref))[0]
        with torch.no_grad():
            d = [
                metric(
                    x.to(device).float().clamp(-1, 1), y.to(device).float().clamp(-1, 1)
                ).item()
                for x, y in zip(images[mode], images[base])
            ]
        row["lpips_vs_" + base] = sum(d) / len(d)
    gpu = getattr(torch, device.type).get_device_name(device).replace(" ", "_")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{gpu}_quant.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"gpu": gpu, "args": vars(args), "rows": rows}, f, indent=1)
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != "quant"}))
    print("wrote", path)


if __name__ == "__main__":
    main()
