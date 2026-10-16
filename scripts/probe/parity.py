"""Parity of the KohakuMoss model and pipeline against the official implementation.

Stages (all write into one JSON report):
  blocks : per-block / output error vs the official fp32 forward, ours fp32 / bf16 /
           fp16 (+fold) and official bf16, at several resolutions and sigmas
  pack   : packed multi-sample forward == per-sample forwards
  fold   : fold exactness on the real weights (gate and per-layer proj modes)
  images : 1024^2, 50 steps, CFG 3, official loop vs MossPipeline (per-step latent
           error, decoded-image error, LPIPS)

    python scripts/probe/parity.py --weights <root>/base --embeds runs/probe/embeds.pt \
        --out runs/probe/parity --ref-device cuda:0 --our-device cuda:1
"""

import argparse
import gc
import json
from pathlib import Path

import lpips
import torch

from kohakumoss.diffusion.sampling import sample_latents, unpack
from kohakumoss.encoders.vae import FluxVae
from kohakumoss.inference.pipeline import to_pil
from kohakumoss.inference.reference import official_module, rel_error
from kohakumoss.model import load_backbone
from kohakumoss.model.numerics import fold, unfold
from kohakumoss.types.seqinfo import build_seqinfo

RESOLUTIONS = ((512, 512), (1024, 1024))
SIGMAS = (0.9, 0.5, 0.1)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def official_transformer(mod, weights, dtype, device):
    t = mod.MOSSImageTransformer.from_pretrained(
        str(Path(weights) / "transformer"), torch_dtype=dtype
    )
    return t.to(device).eval().requires_grad_(False)


def block_hooks(layers, store):
    handles = []
    for i, layer in enumerate(layers):

        def hook(module, args, output, _i=i):
            store[_i] = output.detach().float().cpu()

        handles.append(layer.register_forward_hook(hook))
    return handles


def official_forward(ref, latent, sigma, text, skip=None):
    out = ref(latent, sigma * 1000.0, text[None], skip_layers=skip, return_dict=False)[
        0
    ]
    return out.float()


def our_forward(model, latent, sigma, text, skip=frozenset()):
    _, c, h, w = latent.shape
    dev = next(model.parameters()).device
    seq = build_seqinfo(torch.tensor([[h, w]]), torch.tensor([text.shape[0]])).to(dev)
    img = latent[0].flatten(1).T.to(dev)
    out = model.bind_forward(skip)(img, text.to(dev), sigma.to(dev), seq).float()
    return out.T.reshape(1, c, h, w)


def stage_blocks(args, mod, data, report):
    ref_dev, our_dev = torch.device(args.ref_device), torch.device(args.our_device)
    cases = []
    for height, width in RESOLUTIONS:
        for p, text in enumerate(data["embeds"]):
            for sigma in SIGMAS:
                g = torch.Generator().manual_seed(100 + p)
                lat = torch.randn((1, 128, height // 16, width // 16), generator=g)
                cases.append((height, width, p, sigma, lat, text))
    ref = official_transformer(mod, args.weights, torch.float32, ref_dev)
    ref_blocks, ref_out = {}, []
    ref_store = {}
    handles = block_hooks(ref.layers, ref_store)
    with torch.inference_mode():
        for _, _, _, sigma, lat, text in cases:
            s = torch.tensor([sigma], device=ref_dev)
            ref_out.append(
                official_forward(ref, lat.to(ref_dev), s, text.to(ref_dev)).cpu()
            )
            ref_blocks[len(ref_out) - 1] = {i: v[0] for i, v in ref_store.items()}
    for h in handles:
        h.remove()
    del ref
    free()
    rows = {}
    for name, dtype, fold_c in (
        ("ours_fp32", torch.float32, None),
        ("ours_bf16", torch.bfloat16, None),
        ("ours_fp16", torch.float16, None),
        ("ours_fp16_fold", torch.float16, args.fold),
    ):
        model, _ = load_backbone(args.weights, device=our_dev, dtype=dtype)
        model.requires_grad_(False)
        if fold_c:
            fold(model, fold_c)
        store = {}
        handles = block_hooks(model.layers, store)
        out_rows = []
        with torch.inference_mode():
            for idx, (height, width, p, sigma, lat, text) in enumerate(cases):
                out = our_forward(
                    model, lat, torch.tensor([sigma]), text.float().to(dtype)
                )
                fold_div = model.fold_scales[0] if model.fold_scales else 1.0
                per_block = [
                    rel_error(store[i] / fold_div, ref_blocks[idx][i])
                    for i in sorted(store)
                ]
                out_rows.append(
                    {
                        "res": [height, width],
                        "prompt": p,
                        "sigma": sigma,
                        "out_rel": rel_error(out.cpu(), ref_out[idx]),
                        "finite": bool(torch.isfinite(out).all()),
                        "block_rel_max": max(per_block),
                        "block_rel": per_block,
                    }
                )
                print(name, {k: v for k, v in out_rows[-1].items() if k != "block_rel"})
        for h in handles:
            h.remove()
        rows[name] = out_rows
        del model
        free()
    with torch.inference_mode():
        ref16 = official_transformer(mod, args.weights, torch.bfloat16, our_dev)
        rows["official_bf16"] = [
            {
                "res": [c[0], c[1]],
                "prompt": c[2],
                "sigma": c[3],
                "out_rel": rel_error(
                    official_forward(
                        ref16,
                        c[4].to(our_dev, torch.bfloat16),
                        torch.tensor([c[3]], device=our_dev),
                        c[5].to(our_dev, torch.bfloat16),
                    ).cpu(),
                    ref_out[i],
                ),
            }
            for i, c in enumerate(cases)
        ]
    del ref16
    free()
    report["blocks"] = rows
    report["blocks_summary"] = {
        k: max(r["out_rel"] for r in v) for k, v in rows.items()
    }


def stage_pack_fold(args, data, report):
    dev = torch.device(args.our_device)
    model, _ = load_backbone(args.weights, device=dev, dtype=torch.float32)
    model.requires_grad_(False)
    hws = [(32, 32), (48, 40), (24, 64)]
    texts = [t.to(dev) for t in data["embeds"]]
    g = torch.Generator().manual_seed(7)
    lats = [torch.randn((1, 128, h, w), generator=g) for h, w in hws]
    sigma = torch.tensor([0.8, 0.4, 0.2])
    with torch.inference_mode():
        seq = build_seqinfo(
            torch.tensor(hws), torch.tensor([t.shape[0] for t in texts])
        )
        img = torch.cat([z[0].flatten(1).T for z in lats]).to(dev)
        packed = model(img, torch.cat(texts), sigma.to(dev), seq.to(dev)).float().cpu()
        loop = torch.cat(
            [
                our_forward(model, z, sigma[i : i + 1], texts[i])[0].flatten(1).T.cpu()
                for i, z in enumerate(lats)
            ]
        )
        report["pack"] = {"packed_vs_loop_rel": rel_error(packed, loop)}
        base = our_forward(model, lats[0], sigma[:1], texts[0]).cpu()
        fold_rows = {}
        for label, c, via in (
            ("gate_2^-8", 2.0**-8, "gate"),
            (
                "proj_layers",
                [2.0 ** -(4 + (i % 5)) for i in range(model.cfg.depth)],
                "proj",
            ),
        ):
            fold(model, c, via)
            folded = our_forward(model, lats[0], sigma[:1], texts[0]).cpu()
            unfold(model)
            fold_rows[label] = rel_error(folded, base)
        report["fold"] = fold_rows
    print(report["pack"], report["fold"])
    del model
    free()


def official_loop(mod, ref, latents, pos, neg, steps, cfg, skip, record):
    """The official MOSSImagePipeline denoising loop, recording latents per step."""
    sched = mod.MOSSImageScheduler()
    sched.set_timesteps(steps, device=latents.device)
    x = latents.float()
    for t in sched.timesteps:
        v = ref(x, t, pos, return_dict=False)[0].float()
        u = ref(x, t, neg, skip_layers=skip, return_dict=False)[0].float()
        x = sched.step(u + cfg * (v - u), t, x, return_dict=False)[0]
        record.append(x.cpu())
    return x


def official_decode(vae, z):
    """The decode lines of MOSSImagePipeline.__call__, verbatim."""
    z = z.to(dtype=vae.dtype)
    bn = vae.bn
    eps = vae.config.batch_norm_eps
    z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(
        z
    ).view(1, -1, 1, 1)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    z = z.reshape(b, c // 4, 2 * h, 2 * w)
    return vae.decode(z, return_dict=False)[0].clamp(-1, 1).float()


def stage_images(args, mod, data, report):
    ref_dev, our_dev = torch.device(args.ref_device), torch.device(args.our_device)
    dtype = torch.bfloat16
    ref = official_transformer(mod, args.weights, dtype, ref_dev)
    ours, _ = load_backbone(args.weights, device=our_dev, dtype=dtype)
    ours.requires_grad_(False)
    vae = FluxVae.from_pretrained(args.weights, our_dev, torch.float32)
    metric = lpips.LPIPS(net="alex").to(our_dev)
    out_dir = Path(args.out)
    rows, z0_bank = [], {}
    for p, text in enumerate(data["embeds"]):
        seed = 42 + p
        h = w = 1024 // 16
        g = torch.Generator(ref_dev).manual_seed(seed)
        noise = torch.randn((1, 128, h, w), generator=g, device=ref_dev)
        ref_steps, our_steps = [], []
        with torch.inference_mode():
            ref_x = official_loop(
                mod,
                ref,
                noise,
                text[None].to(ref_dev, dtype),
                data["empty"][None].to(ref_dev, dtype),
                50,
                3.0,
                [22],
                ref_steps,
            )
            res = sample_latents(
                ours,
                [text.to(our_dev, dtype)],
                [data["empty"].to(our_dev, dtype)],
                (1024, 1024),
                50,
                3.0,
                5.0,
                None,
                [seed],
                our_dev,
                "packed",
                lambda i, x: our_steps.append(
                    unpack(x, torch.tensor([[h, w]]))[0].cpu()
                ),
            )
            ours_x = unpack(res.latents, res.hw)[0]
            img_ref = vae.decode(ref_x.to(our_dev))
            img_ours = vae.decode(ours_x)
            vae_rel = rel_error(img_ref, official_decode(vae.vae, ref_x.to(our_dev)))
            lp = float(metric(img_ref, img_ours).mean())
        to_pil(img_ref).save(out_dir / f"official_{p}.png")
        to_pil(img_ours).save(out_dir / f"ours_{p}.png")
        z0_bank[(h, w)] = ours_x.cpu()
        step_err = [rel_error(a, b) for a, b in zip(our_steps, ref_steps)]
        rows.append(
            {
                "prompt": p,
                "seed": seed,
                "steps": len(step_err),
                "latent_rel_final": rel_error(ours_x, ref_x),
                "latent_rel_per_step": step_err,
                "image_rel": rel_error(img_ours, img_ref),
                "lpips": lp,
                "vae_decode_vs_official_rel": vae_rel,
            }
        )
        print(
            {k: v for k, v in rows[-1].items() if k != "latent_rel_per_step"},
            flush=True,
        )
    report["images"] = rows
    torch.save(z0_bank, out_dir / "z0_bank.pt")
    del ref, ours, vae
    free()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--embeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-device", default="cuda:0")
    ap.add_argument("--our-device", default="cuda:1")
    ap.add_argument("--fold", type=float, default=2.0**-4)
    ap.add_argument("--stages", default="blocks,pack,images")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.embeds)
    mod = official_module(args.weights)
    report = {"fold": args.fold}
    stages = args.stages.split(",")
    if "pack" in stages:
        stage_pack_fold(args, data, report)
    if "blocks" in stages:
        stage_blocks(args, mod, data, report)
    if "images" in stages:
        stage_images(args, mod, data, report)
    (out / "parity.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in ("blocks",)}
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
