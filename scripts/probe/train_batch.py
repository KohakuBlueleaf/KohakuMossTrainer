"""Real-weights probe of the training path on single batches of a train config.

Builds the run exactly as the train entry does (model, adapters, kernels, encoders, step,
optimiser bundle), then for each requested batch index dumps per stage: batch layout,
latent statistics per sample (normalised and de-normalised) with a VAE round-trip PSNR,
text embedding statistics and lengths, sigma, target, model output, per-sample loss
(packed forward, per-sample forwards, adapters off), and per-parameter-group grad norms.
`--sweep` compares the adapter gradient across kernel selections and against fp32 /
fp64 references (grad_refs.py).

    python scripts/probe/train_batch.py --config runs/x/cfg.py --out runs/probe/ref \
        --index 0 --sweep --gpu-fp32 eager --cpu-reference fp32 fp64
"""

import argparse
import ast
import json
import math
import runpy
from collections import defaultdict
from pathlib import Path

import torch
from grad_refs import (
    REF_DTYPES,
    dump_attention,
    grad_report,
    group_of,
    op_sweep,
    replay,
    sample_mse,
    split,
)

from kohakumoss.diffusion.objective import TokenNormalizer
from kohakumoss.training.build.kernels import (
    build_adapter_prewarm,
    build_prewarm,
    prewarm_devices,
)
from kohakumoss.training.build.loop import build_batcher, build_optim, build_step
from kohakumoss.training.build.model import build_model
from kohakumoss.training.build.parallel import build_hooks
from kohakumoss.training.knobs import check, collect
from kohakumoss.training.precision import autocast_factory, build_policy
from kohakumoss.training.step import image_tokens_per_sample
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_data import DataConfig, build_loader
from kohakumoss_kernels.tuning.tuner import set_mode as set_tune_mode

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "train" / "moss.py"


def stats(t: torch.Tensor) -> dict:
    """mean / std / absmax / rms / finiteness of a tensor (fp64 reductions)."""
    t = t.detach().double()
    finite = torch.isfinite(t)
    good = t[finite]
    if good.numel() == 0:
        return {"n": t.numel(), "finite": 0}
    return {
        "n": t.numel(),
        "finite": int(finite.sum()),
        "mean": float(good.mean()),
        "std": float(good.std()) if good.numel() > 1 else 0.0,
        "rms": float(good.pow(2).mean().sqrt()),
        "absmax": float(good.abs().max()),
    }


def load_knobs(config: str, sets: list[str]) -> dict:
    """Train-script defaults, then the config file, then `NAME=literal` overrides."""
    g = runpy.run_path(str(TRAIN_SCRIPT), run_name="probe")
    g.update(collect(runpy.run_path(config, run_name="probe")))
    for item in sets:
        name, value = item.split("=", 1)
        g[name.strip()] = ast.literal_eval(value.strip())
    return check(g)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR (dB) of two [-1, 1] images."""
    mse = float((a.double() - b.double()).pow(2).mean())
    return 10 * math.log10(4.0 / max(mse, 1e-20))


def vae_roundtrip(vae, hb) -> list[dict]:
    """Encode each sample's pixels, decode, and compare to the pixels (dB)."""
    if hb.pixels is None:
        return []
    out = []
    dev = vae.mean.device
    for image, (h, w) in zip(hb.pixels, hb.hw.tolist()):
        packed = vae.encode_images([image], torch.tensor([[h, w]]))
        lat = packed.T.reshape(1, -1, h, w)
        rec = vae.decode(lat)
        ref = torch.from_numpy(image).to(dev).permute(2, 0, 1)[None].float() / 127.5 - 1
        raw = lat * vae.std.view(1, -1, 1, 1) + vae.mean.view(1, -1, 1, 1)
        out.append(
            {
                "hw": [h, w],
                "psnr_db": psnr(rec, ref),
                "normalised": stats(lat),
                "raw": stats(raw),
            }
        )
    return out


def per_sample_forward(backbone, autocast, p, img_sizes, txt_sizes, hw, keeps, axes):
    """Each sample alone through the backbone: [N_img, C] in packed order."""
    zs = split(p.z, img_sizes)
    ts = split(p.text, txt_sizes)
    outs = []
    for i in range(len(zs)):
        seq = build_seqinfo(
            hw[i : i + 1],
            torch.tensor([txt_sizes[i]]),
            axes,
            img_keep=None if keeps is None else [keeps[i]],
        ).to(p.z.device)
        with torch.no_grad(), autocast():
            outs.append(backbone(zs[i], ts[i], p.sigma[i : i + 1], seq).float())
    return torch.cat(outs, 0)


def adapter_report(trainable) -> dict:
    by_group = defaultdict(float)
    for name, p in trainable.named_trainable():
        by_group[group_of(name)] += float(p.detach().double().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sorted(by_group.items())}


def probe_batch(k, built, step, bundle, batcher, autocast, hb) -> dict:
    """Every stage of one training batch."""
    rep = {"index": hb.index, "ids": hb.ids.tolist(), "hw": hb.hw.tolist()}
    rep["text_drop"] = hb.text_drop.tolist()
    rep["prompt_chars"] = [len(s) for s in hb.prompts] if hb.prompts else None
    rep["cache_hit"] = {"latents": hb.latents is not None, "text": hb.text is not None}
    rep["vae_roundtrip"] = vae_roundtrip(built.vae, hb) if built.vae else []
    batch = batcher(hb)
    seq = batch.seq
    img_sizes = image_tokens_per_sample(hb)
    txt_sizes = seq.text_len.tolist()
    rep["img_tokens"] = img_sizes
    rep["text_len"] = txt_sizes
    rep["host_text_len"] = hb.text_len.tolist()
    rep["latents"] = [stats(t) for t in split(batch.latents, img_sizes)]
    rep["text"] = [stats(t) for t in split(batch.text, txt_sizes)]
    rep["text_dtype"] = str(batch.text.dtype)
    p = step.prepare(batch, 0)
    rep["sigma"] = p.sigma.tolist()
    rep["target"] = [stats(t) for t in split(p.target, img_sizes)]
    rep["z_sigma"] = [stats(t) for t in split(p.z, img_sizes)]
    denom = step.denominator(batch)
    bundle.zero_grad()
    with autocast():
        pred = step.backbone(p.z, p.text, p.sigma, p.seq)
    loss = step.loss(pred, p, denom)
    rep["pred_dtype"] = str(pred.dtype)
    rep["pred"] = [stats(t) for t in split(pred, img_sizes)]
    rep["loss"] = float(loss.detach())
    rep["mse_per_sample"] = sample_mse(pred.detach(), p.target, img_sizes)
    rep["zero_pred_mse_per_sample"] = sample_mse(
        torch.zeros_like(p.target), p.target, img_sizes
    )
    bundle.backward(loss)
    rep["grads"] = grad_report(built.trainable)
    bundle.zero_grad()
    alone = per_sample_forward(
        step.backbone,
        autocast,
        p,
        img_sizes,
        txt_sizes,
        hb.hw,
        hb.img_keep,
        built.backbone.cfg.rope_axes,
    )
    rep["mse_per_sample_alone"] = sample_mse(alone, p.target, img_sizes)
    rep["packed_vs_alone_rel"] = float(
        (alone - pred.detach().float()).norm() / alone.norm().clamp_min(1e-30)
    )
    multiplier = getattr(built.trainable, "multiplier", None)
    if multiplier is not None:
        built.trainable.set_multiplier(0.0)
        with torch.no_grad(), autocast():
            base = step.backbone(p.z, p.text, p.sigma, p.seq).float()
        built.trainable.set_multiplier(multiplier)
        rep["mse_per_sample_adapters_off"] = sample_mse(base, p.target, img_sizes)
        rep["adapter_effect_rel"] = float(
            (base - pred.detach().float()).norm() / base.norm().clamp_min(1e-30)
        )
    return rep


def dump_step_batch(built, step, batcher, hb, path: Path) -> None:
    """One training step's exact inputs (noised latents, target, text, sigma, layout,
    weights, loss denominator) and the adapter values, for a standalone repro."""
    batch = batcher(hb)
    p = step.prepare(batch, 0)
    seq = p.seq
    torch.save(
        {
            "ids": hb.ids.cpu(),
            "hw": seq.hw.cpu(),
            "text_len": seq.text_len.cpu(),
            "img_keep": None if hb.img_keep is None else list(hb.img_keep),
            "z": p.z.detach().cpu(),
            "target": p.target.detach().cpu(),
            "text": p.text.detach().cpu(),
            "sigma": p.sigma.detach().cpu(),
            "weight": p.weight.detach().cpu(),
            "denom": int(step.denominator(batch)),
            "adapters": {
                name: q.detach().cpu() for name, q in built.trainable.named_trainable()
            },
        },
        path,
    )
    print("dumped", path, flush=True)


BRIEF = (
    "index",
    "sigma",
    "loss",
    "mse_per_sample",
    "mse_per_sample_alone",
    "mse_per_sample_adapters_off",
    "zero_pred_mse_per_sample",
    "packed_vs_alone_rel",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", type=int, nargs="+", default=[0])
    ap.add_argument("--set", action="append", default=[], help="NAME=python-literal")
    ap.add_argument("--sweep", action="store_true", help="kernel backend sweep")
    ap.add_argument("--per-op", action="store_true", help="sweep each op alone")
    ap.add_argument(
        "--draws", type=int, default=1, help="sweep gradients averaged over N draws"
    )
    ap.add_argument(
        "--official",
        action="store_true",
        help="also the official MOSS transformer in bf16 autocast with the same LoRA",
    )
    ap.add_argument(
        "--gpu-fp32",
        nargs="*",
        default=[],
        choices=("eager", "default"),
        help="GPU fp32-compute references over the bf16 checkpoint weights",
    )
    ap.add_argument(
        "--emulate",
        nargs="*",
        default=[],
        help="GPU fp32 runs with bf16 roundings, each a '+'-joined set of: inputs, "
        "linears, stream_fwd, stream_bwd, rms_norm, rms_norm_mod, qk_norm_rope, "
        "attention_varlen, situ_glu, gated_residual",
    )
    ap.add_argument(
        "--cpu-reference",
        nargs="*",
        default=[],
        choices=tuple(REF_DTYPES),
        help="CPU gradient references (fp32 / fp64)",
    )
    ap.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="raise at the first backward node that produces NaN (forward traceback)",
    )
    ap.add_argument(
        "--dump-attention",
        type=int,
        default=None,
        help="save this block's attention inputs and output gradient per batch",
    )
    ap.add_argument(
        "--dump-batch",
        action="store_true",
        help="save each batch's prepared step inputs and adapter values (batch_<i>.pt) "
        "instead of probing it",
    )
    ap.add_argument(
        "--replay",
        action="store_true",
        help="advance the step's sigma / noise generators through every earlier batch "
        "first, so batch i gets the draws of step i + 1 of a run",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    k = load_knobs(args.config, args.set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", k["DEVICES"][0])
    torch.cuda.set_device(device)
    torch.backends.cuda.preferred_linalg_library("cusolver")
    set_tune_mode(k["TUNE"])
    torch.manual_seed(k["SEED"])
    policy = build_policy(
        k["PRECISION"], k["KEEP_FP32"], k["FP16_FIX"], k["FP16_FIX_KWARGS"]
    )
    built = build_model(k, device, policy)
    warm = prewarm_devices(k, device)
    build_prewarm(k, built.backbone, policy.compute)(warm)
    build_adapter_prewarm(k, built, policy.compute)(warm)
    torch.manual_seed(k["SEED"])
    hooks = build_hooks("none", built, policy.compute, 1)
    autocast = autocast_factory(policy, device)
    normalizer = TokenNormalizer(hooks.reduce, hooks.world_factor)
    step = build_step(k, built, normalizer, autocast, device)
    bundle, _ = build_optim(k, built, device, policy, hooks)
    dataset = build_loader(DataConfig.from_knobs(k)).dataset
    batcher = build_batcher(k, built, device)
    report = {
        "knobs": json.loads(json.dumps(k, default=str)),
        "adapter_norms_init": adapter_report(built.trainable),
        "batches": [],
    }
    if args.sweep:
        if args.replay:
            replay(step, dataset, 0, args.index[0], built.backbone.cfg, device)
        hb = dataset.materialize(args.index[0])
        report["sweep_batch"] = {"ids": hb.ids.tolist(), "hw": hb.hw.tolist()}
        report["sweep_batch"]["img_tokens"] = image_tokens_per_sample(hb)
        opts = {
            "per_op": args.per_op,
            "gpu_fp32": args.gpu_fp32,
            "emulate": args.emulate,
            "draws": args.draws,
            "official": args.official,
            "cpu": args.cpu_reference,
        }
        report["sweep"] = op_sweep(
            k, built, step, batcher, autocast, bundle, hb, device, policy.compute, opts
        )
        (out / "report.json").write_text(json.dumps(report, indent=1), "utf-8")
        return
    dump = None
    if args.dump_attention is not None:
        dump = dump_attention(built.backbone, args.dump_attention)
    done = 0
    for index in args.index:
        if args.replay:
            done = replay(step, dataset, done, index, built.backbone.cfg, device)
        hb = dataset.materialize(index)
        if args.dump_batch:
            dump_step_batch(built, step, batcher, hb, out / f"batch_{index}.pt")
            done = index + 1
            continue
        rep = probe_batch(k, built, step, bundle, batcher, autocast, hb)
        done = index + 1
        if dump is not None:
            torch.save(dump.saved, out / f"attn_{args.dump_attention}_{index}.pt")
            dump.records = []
        report["batches"].append(rep)
        brief = {key: rep[key] for key in BRIEF if key in rep}
        brief["grad_finite_norm"] = rep["grads"]["finite_norm"]
        brief["grad_nonfinite_params"] = rep["grads"]["nonfinite_params"]
        brief["nonfinite_by_layer"] = rep["grads"]["nonfinite_by_layer"]
        print(json.dumps(brief), flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=1), "utf-8")
    torch.cuda.synchronize()
    report["vram_peak_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
    (out / "report.json").write_text(json.dumps(report, indent=1), "utf-8")


if __name__ == "__main__":
    main()
