"""Record one 50-step CFG sampling trajectory (official loop or ours) for later comparison.

    python scripts/probe/trajectory.py --impl official --dtype fp32 --prompt 2 \
        --weights <root>/base --embeds runs/probe/embeds.pt --out runs/probe/traj
    python scripts/probe/trajectory.py --compare runs/probe/traj
"""

import argparse
import itertools
import json
from pathlib import Path

import lpips
import torch

from kohakumoss.diffusion.sampling import sample_latents, unpack
from kohakumoss.encoders.vae import FluxVae
from kohakumoss.inference.pipeline import to_pil
from kohakumoss.inference.reference import official_module, rel_error
from kohakumoss.model import load_backbone
from kohakumoss.model.numerics import SaturationCounters, apply_fp16_fix

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def official_traj(args, dtype, text, empty, noise):
    mod = official_module(args.weights)
    ref = mod.MOSSImageTransformer.from_pretrained(
        str(Path(args.weights) / "transformer"), torch_dtype=dtype
    )
    ref = ref.to(noise.device).eval().requires_grad_(False)
    sched = mod.MOSSImageScheduler()
    sched.set_timesteps(args.steps, device=noise.device)
    x, steps = noise.float(), []
    pos, neg = text[None].to(noise.device, dtype), empty[None].to(noise.device, dtype)
    for t in sched.timesteps:
        v = ref(x, t, pos, return_dict=False)[0].float()
        u = ref(x, t, neg, skip_layers=[22], return_dict=False)[0].float()
        x = sched.step(u + args.cfg * (v - u), t, x, return_dict=False)[0]
        steps.append(x.cpu())
    return steps


def ours_traj(args, dtype, text, empty, noise):
    dev = noise.device
    model, _ = load_backbone(args.transformer or args.weights, device=dev, dtype=dtype)
    model.requires_grad_(False)
    apply_fp16_fix(model, args.fp16_fix, **json.loads(args.fp16_fix_kwargs))
    h, w = noise.shape[-2:]
    sat = SaturationCounters(model, threshold=65504.0)
    hw = torch.tensor([[h, w]])
    steps = []
    sample_latents(
        model,
        [text.to(dev, dtype)],
        [empty.to(dev, dtype)],
        (16 * h, 16 * w),
        args.steps,
        args.cfg,
        5.0,
        None,
        [args.seed],
        dev,
        args.cfg_mode,
        lambda i, x: steps.append(unpack(x, hw)[0].cpu()),
    )
    stats = sat.read()
    print(
        "stream maxabs",
        max(v["maxabs"] for v in stats.values()),
        "values >= fp16 max",
        sum(v["over"] for v in stats.values()),
        "final finite",
        bool(torch.isfinite(steps[-1]).all()),
    )
    return steps


def compare(folder: Path) -> None:
    runs = {p.stem: torch.load(p) for p in sorted(folder.glob("*.pt"))}
    out = {}
    for a, b in itertools.combinations(sorted(runs), 2):
        sa, sb = runs[a], runs[b]
        curve = [rel_error(x, y) for x, y in zip(sa, sb)]
        out[f"{a}__vs__{b}"] = {
            "final": curve[-1],
            "at": {i: curve[i] for i in (0, 9, 19, 29, 39, 49) if i < len(curve)},
        }
    (folder / "compare.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def decode_compare(folder: Path, ref: str, weights: str) -> None:
    """Decode every run's final latent, save PNGs, LPIPS + image error vs run `ref`."""
    dev = torch.device("cuda")
    vae = FluxVae.from_pretrained(weights, dev, torch.float32)
    metric = lpips.LPIPS(net="alex").to(dev)
    images = {}
    with torch.inference_mode():
        for p in sorted(folder.glob("*.pt")):
            final = torch.load(p)[-1].to(dev)
            images[p.stem] = vae.decode(final)
            to_pil(images[p.stem]).save(folder / f"{p.stem}.png")
        base = images[ref]
        out = {
            name: {
                "lpips": float(metric(img, base).mean()),
                "image_rel": rel_error(img, base),
                "finite": bool(torch.isfinite(img).all()),
            }
            for name, img in images.items()
            if name != ref
        }
    (folder / f"images_vs_{ref}.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8"
    )
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=("official", "ours"))
    ap.add_argument("--dtype", default="fp32", choices=tuple(DTYPES))
    ap.add_argument("--transformer", default="", help="DiT override (fp16-fix file)")
    ap.add_argument("--fp16-fix", default="none")
    ap.add_argument("--fp16-fix-kwargs", default="{}")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cfg-mode", default="packed")
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--weights")
    ap.add_argument("--embeds")
    ap.add_argument("--out")
    ap.add_argument("--compare", default="")
    ap.add_argument(
        "--decode-ref", default="", help="with --compare: LPIPS vs this run"
    )
    args = ap.parse_args()
    if args.compare:
        compare(Path(args.compare))
        if args.decode_ref:
            decode_compare(Path(args.compare), args.decode_ref, args.weights)
        return
    data = torch.load(args.embeds)
    args.seed = 42 + args.prompt if args.seed is None else args.seed
    dev = torch.device("cuda")
    h = args.size // 16
    g = torch.Generator(dev).manual_seed(args.seed)
    noise = torch.randn((1, 128, h, h), generator=g, device=dev)
    fn = official_traj if args.impl == "official" else ours_traj
    with torch.inference_mode():
        steps = fn(
            args, DTYPES[args.dtype], data["embeds"][args.prompt], data["empty"], noise
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.impl}_{args.dtype}_{args.fp16_fix}"
    torch.save(steps, out / f"{tag}.pt")
    print("saved", tag, len(steps))


if __name__ == "__main__":
    main()
