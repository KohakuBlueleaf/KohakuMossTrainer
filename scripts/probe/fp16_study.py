"""fp16-fix study of one MOSS-Image variant on the real weights, fp32 as the reference.

Phases, every number and image to OUT: text embeddings (bf16 / fp16 vs an fp32 host
reference); per-block maxima of the original and the folded fp32 model (random-latent
probe grid and real sampling trajectories); stock vs exact eps per c; plain fp16 and
bf16 on the official weights (non-finite rate, first overflowing block, decoded
examples); the c sweep (headroom, fp16 vs fp32 latents / LPIPS / PSNR, fp16 subnormal /
zero counts of the scaled weights and the stream); the whole pipeline in fp16 at the
chosen c; the VAE decode in bf16 / fp16.

    kogine run scripts/probe/fp16_study.py --set ROOT=<root>/base --set OUT=runs/fp16/base
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from kohakumoss.encoders.text import QwenTextEncoder, tokenizer_dir
from kohakumoss.encoders.vae import FluxVae
from kohakumoss.inference import MossPipeline
from kohakumoss.inference.pipeline import apply_tune, sampling_autocast
from kohakumoss.logging.log import get_logger, setup
from kohakumoss.model import load_backbone, save_single
from kohakumoss.model.backbone import compute_dtype
from kohakumoss.model.numerics import choose_factors, fold, unfold
from kohakumoss.model.numerics.fold import _row_scales, _scales
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_common.knobs import check_script

log = get_logger("scripts.fp16_study")

ROOT = ""  # variant pipeline dir with transformer/
TE_VAE_ROOT = ""  # pipeline dir holding text_encoder/, tokenizer/, vae/ ("" = ROOT)
OUT = ""  # output dir: report.json, images/, latents/
WORK = "/dev/shm/kohakumoss-rw-fp16"  # swept fp16 checkpoints (one at a time)
DEVICE = "cuda"
PROMPTS = (
    "A photograph of a tabby cat sitting on a sunlit windowsill, shallow depth of field.",
    "An isometric illustration of a small floating island with a lighthouse, waterfalls "
    "pouring off the edges into clouds, soft pastel colors, highly detailed.",
    "A wooden sign on a forest trail that reads 'KOHAKU' in carved letters, morning fog.",
    "An anime-style illustration of a girl reading a book in a quiet library, warm light.",
    "A studio product photo of a ceramic teapot on a marble table, soft shadows.",
    "A busy night market street in the rain, neon signs reflected on wet pavement.",
    "An oil painting of a mountain lake at sunset with a small red boat.",
    "A detailed pencil sketch of an old steam locomotive crossing a stone bridge.",
)
SIGMAS = (1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.02)
RESOLUTIONS = ((512, 512), (1024, 1024), (768, 1344))
C_SWEEP = (1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 256)
CHOSEN_C = 0.0  # c of the whole-pipeline fp16 run (0 = choose_factors on the probe)
SWEEP_PROMPTS = 4  # sampled per c (the first N of PROMPTS)
EXAMPLES = 2  # plain-fp16 decoded examples
SAMPLE = {"sizes": (1024, 1024), "steps": 50, "cfg": 3.0, "seeds": [11], "batch": 1}
EPS_CASES = (
    ((1024, 1024), 0.9),
    ((1024, 1024), 0.3),
    ((1024, 1024), 0.02),
    ((512, 512), 0.5),
)
TARGET = 2.0**12  # max |x| choose_factors aims for (scripts/export/fp16_fix.py)
PROBE_KERNELS = {  # fp32 probe model: composed linears so the proj hooks see o / ffn_*
    "linear_qk_rope": "eager",
    "linear_situ": "eager",
    "linear_gated_residual": "eager",
}
FP16_MAX = 65504.0
FP16_TINY = 2.0**-14  # smallest normal fp16


class Peaks:
    """Running max |x| per (site, block); per-forward non-finite blocks; fp16 stream
    subnormal / zero / total counts per block."""

    def __init__(self, flags: bool):
        self.flags = flags
        self.peak = defaultdict(float)
        self.bad = {}
        self.sub = defaultdict(lambda: [0, 0, 0])

    def put(self, site: str, i: int, x: torch.Tensor) -> None:
        xa = x.detach().abs()
        finite = torch.isfinite(xa)
        v = float(torch.where(finite, xa, torch.zeros_like(xa)).max())
        self.peak[f"{site}.{i}"] = max(self.peak[f"{site}.{i}"], v)
        if not self.flags:
            return
        if not bool(finite.all()):
            self.bad.setdefault(site, set()).add(i)
        if site == "stream":
            s = self.sub[i]
            s[0] += int(((xa > 0) & (xa < FP16_TINY)).sum())
            s[1] += int((xa == 0).sum())
            s[2] += xa.numel()

    def site_max(self) -> dict:
        out = defaultdict(float)
        for k, v in self.peak.items():
            out[k.split(".")[0]] = max(out[k.split(".")[0]], v)
        return dict(out)

    def per_block(self, depth: int, tdepth: int) -> dict:
        sites = sorted({k.split(".")[0] for k in self.peak})
        n = {s: tdepth if s == "text_stream" else depth for s in sites}
        return {s: [self.peak.get(f"{s}.{i}", 0.0) for i in range(n[s])] for s in sites}

    def stream_sub(self) -> dict:
        tot = [sum(v[j] for v in self.sub.values()) for j in range(3)]
        return {
            "subnormal_frac": tot[0] / max(tot[2], 1),
            "zero_frac": tot[1] / max(tot[2], 1),
            "per_block": {i: v for i, v in sorted(self.sub.items())},
        }


def hook(model, rec: Peaks, logits: bool, seq_box: dict) -> list:
    """Forward hooks (streams, proj outputs) and, with `logits`, a _qk wrapper per block;
    returns what `unhook` undoes."""
    handles, saved = [], []

    def out(site, i):
        return lambda m, a, o: rec.put(site, i, o)

    for i, block in enumerate(model.text_embedder.blocks):
        handles.append(block.register_forward_hook(out("text_stream", i)))
    for i, block in enumerate(model.layers):
        handles.append(block.register_forward_hook(out("stream", i)))
        for name in ("ffn_up_gate", "ffn_down", "o"):
            handles.append(block.proj[name].register_forward_hook(out(name, i)))
        if not logits:
            continue
        inner = block._qk
        saved.append((block, inner))

        def qk(*args, _inner=inner, _i=i):
            q, k = _inner(*args)
            cu = seq_box["seq"].attn_joint.cu_seqlens.tolist()
            best = q.new_zeros((), dtype=torch.float32)
            for s, e in zip(cu[:-1], cu[1:]):
                m = torch.einsum("thd,uhd->htu", q[s:e].float(), k[s:e].float())
                best = torch.maximum(best, m.abs().amax() * q.shape[-1] ** -0.5)
            rec.put("logit", _i, best)
            return q, k

        block._qk = qk
    return [handles, saved]


def unhook(state: list) -> None:
    handles, saved = state
    for h in handles:
        h.remove()
    for block, inner in saved:
        block._qk = inner


def forwards(model, embeds: list, rec: Peaks, logits: bool) -> list[dict]:
    """The probe grid (resolutions x prompts x sigmas, random z0 / noise) through
    `model` under the sampler's autocast; per case the output max, the non-finite flag
    and the blocks with a non-finite value."""
    dev = next(model.parameters()).device
    dt = compute_dtype(model)
    autocast = sampling_autocast(model, dev)
    seq_box, rows = {}, []
    state = hook(model, rec, logits, seq_box)
    with torch.inference_mode(), autocast():
        for height, width in RESOLUTIONS:
            h, w = height // 16, width // 16
            for p, text in enumerate(embeds):
                g = torch.Generator(dev).manual_seed(1234 + p)
                noise = torch.randn((h * w, 128), generator=g, device=dev)
                z0 = torch.randn((h * w, 128), generator=g, device=dev)
                seq = build_seqinfo(
                    torch.tensor([[h, w]]), torch.tensor([text.shape[0]])
                )
                seq_box["seq"] = seq.to(dev)
                for sigma in SIGMAS:
                    rec.bad = {}
                    x = ((1 - sigma) * z0 + sigma * noise).to(dt)
                    s = torch.tensor([sigma], device=dev)
                    y = model(x, text.to(dev, dt), s, seq_box["seq"])
                    bad = sorted(rec.bad.get("stream", ()))
                    rows.append(
                        {
                            "res": [height, width],
                            "prompt": p,
                            "sigma": sigma,
                            "out_max": float(y.float().abs().nan_to_num(0, 0, 0).max()),
                            "nonfinite": not bool(torch.isfinite(y).all()),
                            "first_bad_block": bad[0] if bad else None,
                            "bad_blocks": bad,
                            "bad_text_blocks": sorted(rec.bad.get("text_stream", ())),
                        }
                    )
    unhook(state)
    return rows


def finite_cos(a: torch.Tensor, b: torch.Tensor) -> float | None:
    """Cosine of two tensors in fp32 math; None when either has a non-finite value."""
    if not (bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())):
        return None
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def text_embeddings(te_root: Path) -> dict:
    """PROMPTS + the empty prompt encoded in bf16 and fp16 on the device and in fp32 on
    the host; cos of each against fp32 and of fp16 against bf16."""
    out = {}
    for name in ("bf16", "fp16"):
        te = QwenTextEncoder.from_pretrained(te_root, DEVICE, {"dtype": name})
        embeds = te.encode(list(PROMPTS))
        out[name] = [e.float().cpu() for e in embeds] + [te.empty().float().cpu()]
        del te, embeds
        torch.cuda.empty_cache()
    model_dir = te_root / "text_encoder"
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir(model_dir)))
    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32)
    plan = {"memory": "cpu", "forward": "hf", "layout": "padded"}
    te = QwenTextEncoder(model, tok, "cpu", plan, output_device="cpu")
    embeds = te.encode(list(PROMPTS))
    out["fp32"] = [e.float() for e in embeds] + [te.empty().float()]
    del te, model, embeds
    pairs = {
        "bf16_vs_fp32": ("bf16", "fp32"),
        "fp16_vs_fp32": ("fp16", "fp32"),
        "fp16_vs_bf16": ("fp16", "bf16"),
    }
    cos = {
        k: [finite_cos(x, y) for x, y in zip(out[a], out[b])]
        for k, (a, b) in pairs.items()
    }
    finite = {k: [bool(torch.isfinite(x).all()) for x in v] for k, v in out.items()}
    return {"embeds": out, "cos": cos, "finite": finite}


def sample(model, embeds: list, empty: torch.Tensor, n: int, path: Path) -> list:
    """fp32 latents [1, C, h, w] of the first `n` prompts (SAMPLE) through `model`."""
    dev = torch.device(DEVICE)
    pipe = MossPipeline.from_components(model, None, None, dev)
    lat = []
    for p in range(n):
        res = pipe.generate(
            prompt_embeds=[embeds[p].to(dev)],
            negative_embeds=[empty.to(dev)],
            output="latent",
            progress=False,
            **SAMPLE,
        )
        lat.append(res.latents[0].float().cpu().clone())
    torch.save(lat, path)
    return lat


def decode_all(vae: FluxVae, latents: list) -> list[torch.Tensor]:
    """Pixels [1, 3, H, W] in [-1, 1] on the host; non-finite latents decode as 0."""
    out = []
    with torch.inference_mode():
        for z in latents:
            z = torch.nan_to_num(z, 0.0, 0.0, 0.0).to(DEVICE)
            out.append(vae.decode(z).float().cpu())
    return out


def save_png(px: torch.Tensor, path: Path) -> None:
    img = ((px[0].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255)
    Image.fromarray(img.astype(np.uint8)).save(path)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR in dB of two [-1, 1] images (peak-to-peak 2)."""
    mse = float(((a.float() - b.float()) / 2).pow(2).mean())
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def lpips_of(metric, a: torch.Tensor, b: torch.Tensor) -> float:
    with torch.inference_mode():
        return float(metric(a.to(DEVICE), b.to(DEVICE)).item())


def compare(ref_px, ref_lat, px, lat, metric) -> list[dict]:
    """Per image against the reference: finite latents, latent cos, LPIPS, PSNR."""
    return [
        {
            "finite": bool(torch.isfinite(z).all()),
            "latent_cos": finite_cos(z, rl),
            "lpips": lpips_of(metric, p, rp),
            "psnr": psnr(p, rp),
        }
        for rp, rl, p, z in zip(ref_px, ref_lat, px, lat)
    ]


def mean_of(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def weight_counts(ref, c: float, s: tuple) -> dict:
    """Of the fold-scaled tensors of `ref` (bf16, folded with c, or unfolded at c = 1)
    cast to fp16: elements, nonzeros that become zero (flushed), subnormals and infs;
    summed and per tensor family."""
    params = dict(ref.named_parameters())
    fam = defaultdict(lambda: {"n": 0, "flushed": 0, "subnormal": 0, "inf": 0})
    # the scaled rows, from a fold at c != 1 (at c = 1 every multiplier is 1)
    for name, mult in _row_scales(ref, _scales(ref, 0.5), "gate", s).items():
        t = params[name].detach()
        if isinstance(mult, torch.Tensor):
            t = t[mult.to(t.device) != 1]
        h = t.to(torch.float16)
        parts = name.split(".")
        match parts[0]:
            case "layers":
                key = "layers." + ".".join(parts[2:4])
            case "text_embedder" if parts[1] == "blocks":
                key = "text_embedder.blocks." + parts[3]
            case _:
                key = ".".join(parts[:-1])
        r = fam[key]
        r["n"] += t.numel()
        r["flushed"] += int(((h == 0) & (t != 0)).sum())
        r["subnormal"] += int(((h.abs() > 0) & (h.abs() < FP16_TINY)).sum())
        r["inf"] += int(torch.isinf(h).sum())
    keys = ("n", "flushed", "subnormal", "inf")
    return {
        "families": dict(fam),
        "total": {k: sum(v[k] for v in fam.values()) for k in keys},
    }


def eps_check(model, embeds: list, s: tuple) -> dict:
    """Per c: fp32 output error (relative L2) of the exact-eps and the stock-eps fold
    against the unfolded fp32 model, over EPS_CASES."""
    dev = next(model.parameters()).device
    text = embeds[0].to(dev)
    cases = []
    for (height, width), sigma in EPS_CASES:
        h, w = height // 16, width // 16
        g = torch.Generator().manual_seed(h + int(100 * sigma))
        img = torch.randn((h * w, 128), generator=g).to(dev)
        seq = build_seqinfo(torch.tensor([[h, w]]), torch.tensor([text.shape[0]]))
        t = torch.tensor([sigma], device=dev)
        cases.append((img, seq.to(dev), t, [h, w], sigma))
    out = {}
    with torch.inference_mode():
        refs = [model(img, text, t, seq) for img, seq, t, _, _ in cases]
        for c in C_SWEEP:
            rows = []
            for (img, seq, t, hw, sigma), ref in zip(cases, refs):
                row = {"hw": hw, "sigma": sigma}
                for eps in ("exact", "stock"):
                    fold(model, c, "gate", s, eps)
                    d = model(img, text, t, seq) - ref
                    row[f"{eps}_rel"] = float(d.float().norm() / ref.float().norm())
                    unfold(model)
                rows.append(row)
            out[str(c)] = rows
    return out


def whole_fp16(m, te, vae_root, fix_px, ref_px, ref_lat, metric, out) -> dict:
    """The fixed fp16 transformer with fp16 text embeddings and an fp16 VAE decode,
    against fp32 and against the fixed transformer with bf16 embeddings / fp32 decode.
    """
    n = len(fix_px)
    embeds, empty = te["embeds"]["fp16"][:-1], te["embeds"]["fp16"][-1]
    lat = sample(m, embeds, empty, n, out / "latents" / "whole_fp16.pt")
    vae16 = FluxVae.from_pretrained(vae_root, DEVICE, torch.float16)
    px = decode_all(vae16, lat)
    del vae16
    torch.cuda.empty_cache()
    for p, x in enumerate(px):
        save_png(x, out / "images" / f"whole_fp16_p{p}.png")
    return {
        "vs_fp32": compare(ref_px[:n], ref_lat[:n], px, lat, metric),
        "vs_fix_bf16_te": [
            {"lpips": lpips_of(metric, a, b), "psnr": psnr(a, b)}
            for a, b in zip(px, fix_px)
        ],
    }


def sweep_row(c, s, rec16: Peaks, frows, vs, counts, depth, tdepth) -> dict:
    peak = rec16.site_max()
    top = max(peak.get("stream", 0.0), peak.get("text_stream", 0.0), 1e-30)
    return {
        "c": c,
        "s": list(s),
        "fp16_stream_max": peak.get("stream", 0.0),
        "fp16_text_stream_max": peak.get("text_stream", 0.0),
        "headroom": FP16_MAX / top,
        "nonfinite_forwards": sum(r["nonfinite"] for r in frows),
        "forwards": len(frows),
        "per_block": rec16.per_block(depth, tdepth),
        "stream": rec16.stream_sub(),
        "vs_fp32": vs,
        "mean_lpips": mean_of(vs, "lpips"),
        "mean_psnr": mean_of(vs, "psnr"),
        "mean_latent_cos": mean_of(vs, "latent_cos"),
        "weights": counts,
    }


def main() -> None:
    setup()
    check_script(globals())
    if not ROOT or not OUT:
        raise SystemExit("set ROOT and OUT")
    apply_tune("off")
    root, out = Path(ROOT), Path(OUT)
    te_root = Path(TE_VAE_ROOT or ROOT)
    work = Path(WORK) / out.name
    for d in (out / "images", out / "latents", work):
        d.mkdir(parents=True, exist_ok=True)
    dev = torch.device(DEVICE)
    report = {
        "root": str(root),
        "te_vae_root": str(te_root),
        "prompts": list(PROMPTS),
        "sigmas": list(SIGMAS),
        "resolutions": [list(r) for r in RESOLUTIONS],
        "sample": {
            k: list(v) if isinstance(v, tuple) else v for k, v in SAMPLE.items()
        },
        "c_sweep": list(C_SWEEP),
        "target": TARGET,
    }

    def dump(name: str) -> None:
        (out / name).write_text(json.dumps(report, indent=1, default=str))

    te = text_embeddings(te_root)
    report["text_encoder"] = {"cos": te["cos"], "finite": te["finite"]}
    torch.save(te["embeds"], out / "latents" / "embeds.pt")
    embeds, empty = te["embeds"]["bf16"][:-1], te["embeds"]["bf16"][-1]
    log.info(json.dumps(report["text_encoder"]))
    dump("report.partial.json")

    model, _ = load_backbone(
        root / "transformer", dev, torch.float32, keep_fp32=(), kernels=PROBE_KERNELS
    )
    model.requires_grad_(False)
    depth, tdepth = len(model.layers), len(model.text_embedder.blocks)
    rec = Peaks(False)
    rows = forwards(model, embeds + [empty], rec, logits=True)
    factors = choose_factors({"summary_max": rec.site_max()}, TARGET)
    s = (factors["s_attn"], factors["s_ffn"])
    chosen = CHOSEN_C or factors["c"]
    sampled = Peaks(False)
    state = hook(model, sampled, False, {})
    ref_lat = sample(model, embeds, empty, len(PROMPTS), out / "latents" / "fp32.pt")
    unhook(state)
    fold(model, chosen, "gate", s, "exact")
    folded = Peaks(False)
    forwards(model, embeds + [empty], folded, logits=True)
    unfold(model)
    report["probe"] = {
        "factors": factors,
        "chosen_c": chosen,
        "summary_max": rec.site_max(),
        "sampled_summary_max": sampled.site_max(),
        "folded_summary_max": folded.site_max(),
        "per_block": rec.per_block(depth, tdepth),
        "per_block_sampled": sampled.per_block(depth, tdepth),
        "per_block_folded": folded.per_block(depth, tdepth),
        "rows": rows,
    }
    log.info(json.dumps({"factors": factors, "summary_max": rec.site_max()}))
    report["eps"] = eps_check(model, embeds, s)
    del model
    torch.cuda.empty_cache()
    dump("report.partial.json")

    vae = FluxVae.from_pretrained(te_root, DEVICE, torch.float32)
    metric = lpips.LPIPS(net="alex", verbose=False).to(DEVICE)
    ref_px = decode_all(vae, ref_lat)
    for p, px in enumerate(ref_px):
        save_png(px, out / "images" / f"fp32_p{p}.png")

    plain = {}
    for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        m, _ = load_backbone(root / "transformer", dev, dtype)
        m.requires_grad_(False)
        rec16 = Peaks(True)
        plain[f"{name}_forward_rows"] = forwards(m, embeds + [empty], rec16, False)
        plain[f"{name}_summary_max"] = rec16.site_max()
        plain[f"{name}_per_block"] = rec16.per_block(depth, tdepth)
        plain[f"{name}_stream"] = rec16.stream_sub()
        n = EXAMPLES if name == "fp16" else len(PROMPTS)
        lat = sample(m, embeds, empty, n, out / "latents" / f"official_{name}.pt")
        px = decode_all(vae, lat)
        for p, x in enumerate(px):
            save_png(x, out / "images" / f"official_{name}_p{p}.png")
        plain[f"{name}_vs_fp32"] = compare(ref_px, ref_lat, px, lat, metric)
        del m
        torch.cuda.empty_cache()
        report["plain"] = plain
        dump("report.partial.json")

    ref, _ = load_backbone(root / "transformer", "cpu", torch.bfloat16)
    ref.requires_grad_(False)
    report["weights_c1"] = weight_counts(ref, 1.0, s)
    report["sweep"] = []
    for c in C_SWEEP:
        tag = f"c{int(round(1 / c))}"
        path = work / f"fix_{tag}.safetensors"
        fold(ref, c, "gate", s, "stock")
        counts = weight_counts(ref, c, s)
        save_single(ref, path, dtype=torch.float16, keep_fold=True)
        unfold(ref)
        m, _ = load_backbone(path, dev, torch.float16)
        m.requires_grad_(False)
        rec16 = Peaks(True)
        frows = forwards(m, embeds + [empty], rec16, False)
        lat = sample(m, embeds, empty, SWEEP_PROMPTS, out / "latents" / f"fix_{tag}.pt")
        px = decode_all(vae, lat)
        for p, x in enumerate(px):
            save_png(x, out / "images" / f"fix_{tag}_p{p}.png")
        vs = compare(ref_px[:SWEEP_PROMPTS], ref_lat[:SWEEP_PROMPTS], px, lat, metric)
        row = sweep_row(c, s, rec16, frows, vs, counts, depth, tdepth)
        if c == chosen:
            report["whole_fp16"] = whole_fp16(
                m, te, te_root, px, ref_px, ref_lat, metric, out
            )
        del m
        torch.cuda.empty_cache()
        path.unlink()
        report["sweep"].append(row)
        keys = ("c", "headroom", "nonfinite_forwards", "mean_lpips", "mean_psnr")
        log.info(json.dumps({k: row[k] for k in keys}))
        dump("report.partial.json")
    del ref

    decoded = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        v = FluxVae.from_pretrained(te_root, DEVICE, dtype)
        decoded[name] = decode_all(v, ref_lat)
        del v
        torch.cuda.empty_cache()
    report["vae"] = {
        name: [
            {"psnr_vs_fp32": psnr(a, b), "finite": bool(torch.isfinite(a).all())}
            for a, b in zip(px, ref_px)
        ]
        for name, px in decoded.items()
    }
    report["vae"]["fp16_vs_bf16_psnr"] = [
        psnr(a, b) for a, b in zip(decoded["fp16"], decoded["bf16"])
    ]
    dump("report.json")
    log.info(f"wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
