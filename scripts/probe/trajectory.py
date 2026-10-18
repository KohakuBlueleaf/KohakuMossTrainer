"""Record one 50-step CFG sampling trajectory (official loop or ours) for later comparison.

    kogine run scripts/probe/trajectory.py --set IMPL=official --set PRECISION=fp32 \
        --set PROMPT=2 --set WEIGHTS=<root>/base --set EMBEDS=runs/probe/embeds.pt \
        --set OUT=runs/probe/traj
    kogine run scripts/probe/trajectory.py --set COMPARE=runs/probe/traj
"""

import itertools
import json
from pathlib import Path

import lpips
import torch

from kohakumoss.diffusion.sampling import sample_latents, unpack
from kohakumoss.encoders.vae import FluxVae
from kohakumoss.inference.pipeline import to_pil
from kohakumoss.logging.log import get_logger, setup
from kohakumoss.model import load_backbone
from kohakumoss.model.numerics import SaturationCounters, apply_fp16_fix
from kohakumoss.probe.official import official_module, rel_error
from kohakumoss_common.knobs import check_script

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
log = get_logger("scripts.trajectory")

IMPL = "ours"  # "official" | "ours"
PRECISION = "fp32"  # compute dtype: "fp32" | "bf16" | "fp16"
TRANSFORMER = ""  # DiT override (fp16-fix file); "" = WEIGHTS
FP16_FIX = "none"  # ours: "none" | "weights" | "online" | "fp32_residual"
FP16_FIX_KWARGS = {}  # ours: e.g. {"c": 0.0625}
TAG = ""  # output name; "" = <IMPL>_<PRECISION>_<FP16_FIX>
CFG_MODE = "packed"  # ours: "packed" | "two_pass"
PROMPT = 0  # index into the embeddings file
SEED: int | None = None  # None = 42 + PROMPT
SIZES = [(1024, 1024)]  # (H, W) of the image; one trajectory, so the first entry
NOISE = ""  # saved initial noise [1, 128, H, W] (.pt); "" = drawn from SEED on cuda
STEPS = 50
CFG = 3.0
WEIGHTS = ""  # pipeline dir with transformer/ (and vae/ for DECODE_REF)
EMBEDS = ""  # embeddings file from scripts/probe/embed.py
OUT = ""  # output directory of <TAG>.pt
COMPARE = ""  # directory of recorded runs: compare them instead of recording
DECODE_REF = ""  # with COMPARE: decode every run, LPIPS vs this run


def official_traj(dtype, text, empty, noise):
    mod = official_module(WEIGHTS)
    ref = mod.MOSSImageTransformer.from_pretrained(
        str(Path(WEIGHTS) / "transformer"), torch_dtype=dtype
    )
    ref = ref.to(noise.device).eval().requires_grad_(False)
    sched = mod.MOSSImageScheduler()
    sched.set_timesteps(STEPS, device=noise.device)
    x, steps = noise.float(), []
    pos, neg = text[None].to(noise.device, dtype), empty[None].to(noise.device, dtype)
    for t in sched.timesteps:
        v = ref(x, t, pos, return_dict=False)[0].float()
        u = ref(x, t, neg, skip_layers=[22], return_dict=False)[0].float()
        x = sched.step(u + CFG * (v - u), t, x, return_dict=False)[0]
        steps.append(x.cpu())
    return steps


def ours_traj(dtype, text, empty, noise, seed):
    dev = noise.device
    model, _ = load_backbone(TRANSFORMER or WEIGHTS, device=dev, dtype=dtype)
    model.requires_grad_(False)
    apply_fp16_fix(model, FP16_FIX, **FP16_FIX_KWARGS)
    h, w = noise.shape[-2:]
    sat = SaturationCounters(model, threshold=65504.0)
    hw = torch.tensor([[h, w]])
    steps = []
    sample_latents(
        model,
        [text.to(dev, dtype)],
        [empty.to(dev, dtype)],
        (16 * h, 16 * w),
        STEPS,
        CFG,
        5.0,
        None,
        [seed],
        dev,
        CFG_MODE,
        lambda i, x: steps.append(unpack(x, hw)[0].cpu()),
        noise[0].flatten(1).T,
    )
    stats = sat.read()
    log.info(
        f"stream maxabs {max(v['maxabs'] for v in stats.values())} "
        f"values >= fp16 max {sum(v['over'] for v in stats.values())} "
        f"final finite {bool(torch.isfinite(steps[-1]).all())}"
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
    log.info(json.dumps(out, indent=1))


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
    log.info(json.dumps(out, indent=1))


def main() -> None:
    setup()
    check_script(globals())
    if COMPARE:
        compare(Path(COMPARE))
        if DECODE_REF:
            decode_compare(Path(COMPARE), DECODE_REF, WEIGHTS)
        return
    if IMPL not in ("official", "ours"):
        raise SystemExit(f"IMPL={IMPL!r} not in ('official', 'ours')")
    data = torch.load(EMBEDS)
    seed = 42 + PROMPT if SEED is None else SEED
    dev = torch.device("cuda")
    h, w = SIZES[0][0] // 16, SIZES[0][1] // 16
    if NOISE:
        noise = torch.load(NOISE).to(dev)
    else:
        g = torch.Generator(dev).manual_seed(seed)
        noise = torch.randn((1, 128, h, w), generator=g, device=dev)
    text, empty = data["embeds"][PROMPT], data["empty"]
    with torch.inference_mode():
        if IMPL == "official":
            steps = official_traj(DTYPES[PRECISION], text, empty, noise)
        else:
            steps = ours_traj(DTYPES[PRECISION], text, empty, noise, seed)
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    tag = TAG or f"{IMPL}_{PRECISION}_{FP16_FIX}"
    torch.save(steps, out / f"{tag}.pt")
    log.info(f"saved {tag} {len(steps)}")


if __name__ == "__main__":
    main()
