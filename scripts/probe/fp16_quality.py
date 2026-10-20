"""fp16-fix quality of one MOSS-Image variant: the fixed fp16 transformer vs the official
weights in plain fp16, bf16 and fp32, the study's prompts x SEEDS, per runtime
(KohakuMoss; the official diffusers MOSSImagePipeline; ComfyUI images from the
ComfyUI-venv runner), LPIPS / SSIM / PSNR per image against that runtime's fp32 (with an
adapter: against fp32 + adapter); plus every fix-file value against the official weights,
the study reproduction checks and the model-card diffusers snippet.

    kogine run scripts/probe/fp16_quality.py --set ROOT=<root>/base --set STUDY=<study out> \
        --set FIX_FILE=<repo>/comfyui/moss_image_base_fp16_fixed.safetensors \
        --set FIX_DIR=<repo>/base/transformer --set OUT=<out>
"""

import importlib
import json
import math
import re
from pathlib import Path

import lpips
import numpy as np
import torch
from diffusers import AutoencoderKLFlux2, DiffusionPipeline
from PIL import Image
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

from kohakumoss.encoders.vae import FluxVae
from kohakumoss.inference import MossPipeline
from kohakumoss.inference.pipeline import apply_tune
from kohakumoss.logging.log import get_logger, setup
from kohakumoss.model import load_backbone
from kohakumoss.model.io.load import INDEX, read_fp16_fix
from kohakumoss.probe.image_metrics import psnr, ssim
from kohakumoss.probe.official import official_module
from kohakumoss_common.knobs import check_script

log = get_logger("scripts.fp16_quality")

ROOT = ""  # variant dir with transformer/ (official weights + pipeline_moss_image.py)
PIPE_ROOT = ""  # full official pipeline dir of the variant, for the card snippet
TE_VAE_ROOT = ""  # pipeline dir holding text_encoder/, tokenizer/, vae/ ("" = ROOT)
STUDY = ""  # scripts/probe/fp16_study.py OUT of this variant (prompts, embeds, latents)
FIX_FILE = ""  # ComfyUI fp16-fix single file
FIX_DIR = ""  # diffusers fp16-fix transformer dir
COMFY_DIR = ""  # images of the ComfyUI-venv runner ("" = not scored)
ADAPTERS = ()  # adapter files trained on the official weights (KohakuMoss runtime)
PLAIN_FP16_PROMPTS = (
    0  # prompts of the official-weights plain-fp16 config (0 = every prompt)
)
COMFY_REF = ""  # a Comfy-Org MOSS-Image file the single file's layout must match
OUT = ""  # output dir: quality.json, images/<runtime>/
RUNTIMES = ("kohakumoss", "diffusers")
SEEDS = (11, 12)
DEVICE = "cuda"
SIZE, STEPS, CFG = (1024, 1024), 50, 3.0
PROBE_KERNELS = {  # the study's fp32 model binding (scripts/probe/fp16_study.py)
    "linear_qk_rope": "eager",
    "linear_situ": "eager",
    "linear_gated_residual": "eager",
}


def save_png(px: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = ((px[0].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255)
    Image.fromarray(img.astype(np.uint8)).save(path)


def load_png(path: Path) -> torch.Tensor:
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return torch.from_numpy(a / 127.5 - 1).permute(2, 0, 1)[None]


def score(metric, imgs: dict, ref: str, names: list[str]) -> dict:
    """Per config in `names`: LPIPS / SSIM / PSNR of each image vs config `ref` (same
    key)."""
    out = {}
    for name in names:
        rows = []
        for key, px in imgs[name].items():
            if key not in imgs[ref]:
                continue
            r = imgs[ref][key]
            with torch.inference_mode():
                d = float(metric(px.to(DEVICE), r.to(DEVICE)).item())
            rows.append(
                {"key": key, "lpips": d, "ssim": ssim(px, r), "psnr": psnr(px, r)}
            )
        n = max(len(rows), 1)
        out[name] = {
            "rows": rows,
            **{
                f"mean_{k}": sum(x[k] for x in rows) / n
                for k in ("lpips", "ssim", "psnr")
            },
        }
    return out


def expected_scale(key: str, rows: int, dim: int, fix: dict) -> torch.Tensor:
    """The fold's per-row factor of one official tensor (docs/concepts/numerics.md),
    stated independently of kohakumoss.model.numerics.fold."""
    c, sa, sf = fix["c"], fix["s_attn"], fix["s_ffn"]
    s = torch.ones(rows, dtype=torch.float64)
    writer = (
        key.startswith("video_embedder.")
        or key.startswith("text_embedder.proj.")
        or re.fullmatch(
            r"text_embedder\.blocks\.\d+\.(proj|ffn_down)\.(weight|bias)", key
        )
    )
    if writer:
        s[:] = c
    elif key.startswith("shared_block_modulation."):
        s[2 * dim : 3 * dim] = c / sa
        s[5 * dim : 6 * dim] = c / sf
    elif re.fullmatch(r"layers\.\d+\.proj\.o\.weight", key):
        s[:] = sa
    elif re.fullmatch(r"layers\.\d+\.proj\.ffn_down\.weight", key):
        s[:] = sf
    return s


def value_rows(path: Path, want: dict, dim: int, fix: dict) -> dict:
    """Every tensor of one fix file against official x the stated factor, rounded once
    to fp16 (a fused proj.qkv against official q|k|v): exact-mismatch, flushed and
    overflow counts."""
    out = {"n": 0, "mismatch": 0, "flushed": 0, "overflow": 0, "tensors": 0}
    with safe_open(str(path), "pt") as f:
        for key in f.keys():
            got = f.get_tensor(key)
            if key.endswith(".proj.qkv.weight"):
                base = key[: -len(".qkv.weight")]
                ref = torch.cat([want[f"{base}.{n}.weight"] for n in "qkv"], 0)
            else:
                ref = want[key]
            if key == "rope.bands":
                exp = ref.float()
            else:
                s = expected_scale(key, ref.shape[0], dim, fix)
                s = s.reshape(-1, *([1] * (ref.dim() - 1)))
                exp = torch.from_numpy((ref.double() * s).numpy().astype(np.float16))
            out["n"] += ref.numel()
            out["tensors"] += 1
            out["mismatch"] += int((got.float() != exp.float()).sum())
            out["flushed"] += int(((ref != 0) & (got == 0)).sum())
            out["overflow"] += int(torch.isinf(got).sum())
    return out


def shapes_of(path: Path) -> dict:
    with safe_open(str(path), "pt") as f:
        return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}


def file_checks(root: Path) -> dict:
    """The real fix files: fold records (file and dir agree), every value against the
    official weights x the stated factors, the single file's layout against COMFY_REF,
    the dir's config.json and keys against the official dir."""
    official = root / "transformer"
    index = json.loads((official / INDEX).read_text(encoding="utf-8"))
    want = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(official / shard), "pt") as f:
            want.update({k: f.get_tensor(k) for k in f.keys()})
    rec_file, rec_dir = read_fp16_fix(FIX_FILE), read_fp16_fix(FIX_DIR)
    dim = want["shared_block_modulation.weight"].shape[0] // 6
    fix_dir = Path(FIX_DIR)
    dir_files = sorted(fix_dir.glob("*.safetensors"))
    dir_rows = [value_rows(p, want, dim, rec_dir) for p in dir_files]
    out = {
        "record_file": rec_file,
        "record_dir": rec_dir,
        "records_agree": rec_file == rec_dir,
        "single_values": value_rows(Path(FIX_FILE), want, dim, rec_file),
        "dir_values": {k: sum(r[k] for r in dir_rows) for k in dir_rows[0]},
        "dir_keys_match_official": sorted(k for p in dir_files for k in shapes_of(p))
        == sorted(index["weight_map"]),
        "dir_config_matches_official": json.loads(
            (fix_dir / "config.json").read_text(encoding="utf-8")
        )
        == json.loads((official / "config.json").read_text(encoding="utf-8")),
    }
    if COMFY_REF:
        mine, theirs = shapes_of(Path(FIX_FILE)), shapes_of(Path(COMFY_REF))
        out["comfy_layout"] = {
            "ref": Path(COMFY_REF).name,
            "missing": sorted(set(theirs) - set(mine)),
            "extra": sorted(set(mine) - set(theirs)),
            "shape_diff": sorted(
                k for k in mine if k in theirs and mine[k] != theirs[k]
            ),
        }
    return out


def lora_scores(metric, imgs: dict) -> dict:
    """Per adapter i: the fixed fp16 + adapter vs official bf16 + adapter, the adapter's
    own effect (official bf16 + adapter vs official bf16), and plain fp16 / fixed fp16 /
    bf16 + adapter vs fp32 + adapter where that exists."""
    out = {}
    for name in sorted(n for n in imgs if n.startswith("bf16_lora")):
        i = name[len("bf16_lora") :]
        fix = f"fix_fp16_lora{i}"
        if fix not in imgs:
            continue
        out[f"lora{i}"] = {
            "fix_vs_bf16": score(metric, imgs, name, [fix])[fix],
            "effect_vs_bf16": score(metric, imgs, "bf16", [name])[name],
        }
        ref = f"fp32_lora{i}"
        if ref in imgs:
            names = [n for n in (f"official_fp16_lora{i}", fix, name) if n in imgs]
            out[f"lora{i}"]["vs_fp32"] = score(metric, imgs, ref, names)
    return out


def plain_prompts(n: int) -> int:
    """Prompts of the plain-fp16 config: PLAIN_FP16_PROMPTS, capped at `n` (0 = all)."""
    return min(PLAIN_FP16_PROMPTS, n) if PLAIN_FP16_PROMPTS else n


def max_diff(a: list, b: list) -> float:
    """Largest |a - b| over paired latents (inf when shapes differ or values are not
    finite)."""
    worst = 0.0
    for x, y in zip(a, b):
        if x.shape != y.shape:
            return math.inf
        worst = max(
            worst, float((x.float() - y.float()).abs().nan_to_num(math.inf).max())
        )
    return worst


def generate(pipe, embeds: list, empty: torch.Tensor, n: int) -> dict:
    """fp32 latents [1, C, h, w] of the first `n` prompts x SEEDS, keyed p<i>_s<seed>."""
    dev = torch.device(DEVICE)
    lat = {}
    for p in range(n):
        for seed in SEEDS:
            res = pipe.generate(
                prompt_embeds=[embeds[p].to(dev)],
                negative_embeds=[empty.to(dev)],
                sizes=SIZE,
                steps=STEPS,
                cfg=CFG,
                seeds=[seed],
                batch=1,
                output="latent",
                progress=False,
            )
            lat[f"p{p}_s{seed}"] = res.latents[0].float().cpu().clone()
    return lat


def kohakumoss_runtime(root, te_root, study, out) -> tuple[dict, dict, dict]:
    """KohakuMoss MossPipeline on the study's bf16 embeddings; fp32 VAE decode. Each
    ADAPTERS file is merged into the official weights (fp32, bf16, plain fp16) and the
    fixed fp16 transformer."""
    data = torch.load(study / "latents" / "embeds.pt")
    embeds, empty = data["bf16"][:-1], data["bf16"][-1]
    dev = torch.device(DEVICE)
    configs = (
        ("fp32", root / "transformer", torch.float32, len(embeds)),
        ("bf16", root / "transformer", torch.bfloat16, len(embeds)),
        ("fix_fp16", Path(FIX_FILE), torch.float16, len(embeds)),
        ("fixdir_fp16", Path(FIX_DIR), torch.float16, len(embeds)),
        (
            "official_fp16",
            root / "transformer",
            torch.float16,
            plain_prompts(len(embeds)),
        ),
    )
    latents, adopted = {}, {}
    for name, path, dtype, n in configs:
        kw = (
            {"keep_fp32": (), "kernels": PROBE_KERNELS}
            if dtype == torch.float32
            else {}
        )
        model, _ = load_backbone(path, dev, dtype, **kw)
        model.requires_grad_(False)
        adopted[name] = None if model.fold_scales is None else model.fold_scales[0]
        pipe = MossPipeline.from_components(model, None, None, dev)
        latents[name] = generate(pipe, embeds, empty, n)
        torch.save(latents[name], out / "latents" / f"kohakumoss_{name}.pt")
        del model, pipe
        torch.cuda.empty_cache()
    for i, adapter in enumerate(ADAPTERS):
        for base, path, dtype in (
            ("fp32", root / "transformer", torch.float32),
            ("bf16", root / "transformer", torch.bfloat16),
            ("official_fp16", root / "transformer", torch.float16),
            ("fix_fp16", Path(FIX_FILE), torch.float16),
            ("fixdir_fp16", Path(FIX_DIR), torch.float16),
        ):
            name = f"{base}_lora{i}"
            pipe = MossPipeline.from_pretrained(
                root,
                transformer=path,
                dtype=dtype,
                device=dev,
                adapters=[(adapter, 1.0)],
                adapter_mode="merge",
                text_encoder=False,
                vae=False,
            )
            latents[name] = generate(pipe, embeds, empty, len(embeds))
            torch.save(latents[name], out / "latents" / f"kohakumoss_{name}.pt")
            del pipe
            torch.cuda.empty_cache()
    vae = FluxVae.from_pretrained(te_root, DEVICE, torch.float32)
    imgs = {}
    with torch.inference_mode():
        for name, lat in latents.items():
            imgs[name] = {}
            for key, z in lat.items():
                z = torch.nan_to_num(z, 0.0, 0.0, 0.0).to(DEVICE)
                px = vae.decode(z).float().cpu()
                imgs[name][key] = px
                save_png(px, out / "images" / "kohakumoss" / f"{name}_{key}.png")
    del vae
    torch.cuda.empty_cache()
    s0 = SEEDS[0]
    c = read_fp16_fix(FIX_FILE)["c"]
    first = [f"p{p}_s{s0}" for p in range(len(embeds))]
    repro = {
        "fp32_vs_study": max_diff(
            [latents["fp32"][k] for k in first],
            torch.load(study / "latents" / "fp32.pt"),
        ),
        "bf16_vs_study": max_diff(
            [latents["bf16"][k] for k in first],
            torch.load(study / "latents" / "official_bf16.pt"),
        ),
        "fix_vs_study_sweep": max_diff(
            [latents["fix_fp16"][k] for k in first],
            torch.load(study / "latents" / f"fix_c{int(round(1 / c))}.pt"),
        ),
        "fixdir_vs_fix": max_diff(
            list(latents["fixdir_fp16"].values()), list(latents["fix_fp16"].values())
        ),
        "official_fp16_vs_study": max_diff(
            [latents["official_fp16"][f"p0_s{s0}"]],
            torch.load(study / "latents" / "official_fp16.pt")[:1],
        ),
        "fixdir_lora_vs_fix_lora": {
            f"lora{i}": max_diff(
                list(latents[f"fixdir_fp16_lora{i}"].values()),
                list(latents[f"fix_fp16_lora{i}"].values()),
            )
            for i in range(len(ADAPTERS))
        },
        "fold_adopted": adopted,
        "fix_c": c,
        "nonfinite_latents": {
            name: sum(not bool(torch.isfinite(z).all()) for z in lat.values())
            for name, lat in latents.items()
        },
    }
    return imgs, repro, {"embeds": "study bf16 (KohakuMoss text encoder)"}


def diffusers_images(mod, transformer, vae, enc, n: int, name: str, out) -> dict:
    """The official MOSSImagePipeline around `transformer` and the fp32 VAE."""
    pipe = mod.MOSSImagePipeline(transformer, mod.MOSSImageScheduler(), None, None, vae)
    pipe.set_progress_bar_config(disable=True)
    embeds, mask, neg, nmask = enc
    imgs = {}
    for p in range(n):
        for seed in SEEDS:
            res = pipe(
                prompt_embeds=embeds[p : p + 1],
                prompt_attention_mask=mask[p : p + 1],
                negative_prompt_embeds=neg,
                negative_prompt_attention_mask=nmask,
                height=SIZE[0],
                width=SIZE[1],
                num_inference_steps=STEPS,
                guidance_scale=CFG,
                generator=torch.Generator(DEVICE).manual_seed(seed),
                output_type="pt",
            )
            px = (res.images.float().cpu() * 2 - 1).clamp(-1, 1)
            key = f"p{p}_s{seed}"
            imgs[key] = px
            save_png(px, out / "images" / "diffusers" / f"{name}_{key}.png")
    return imgs


def diffusers_runtime(root, te_root, prompts, out) -> tuple[dict, dict, dict]:
    """Official pipeline code from the weights; prompts encoded once by its own
    encode_prompt (bf16 text encoder), then each transformer in turn."""
    mod = official_module(root)
    dev = torch.device(DEVICE)
    tok = AutoTokenizer.from_pretrained(str(te_root / "tokenizer"))
    te = AutoModel.from_pretrained(str(te_root / "text_encoder"), dtype=torch.bfloat16)
    te = te.eval().to(dev)
    enc_pipe = mod.MOSSImagePipeline(None, mod.MOSSImageScheduler(), te, tok, None)
    embeds, mask = enc_pipe.encode_prompt(list(prompts))
    neg, nmask = enc_pipe.encode_prompt([""])
    enc = (embeds, mask, neg, nmask)
    del enc_pipe, te
    torch.cuda.empty_cache()
    vae = AutoencoderKLFlux2.from_pretrained(
        str(te_root), subfolder="vae", torch_dtype=torch.float32
    )
    vae = vae.eval().to(dev)
    configs = (
        ("fp32", root / "transformer", torch.float32, len(prompts)),
        ("bf16", root / "transformer", torch.bfloat16, len(prompts)),
        ("fix_fp16", Path(FIX_DIR), torch.float16, len(prompts)),
        (
            "official_fp16",
            root / "transformer",
            torch.float16,
            plain_prompts(len(prompts)),
        ),
    )
    imgs = {}
    for name, path, dtype, n in configs:
        tr = mod.MOSSImageTransformer.from_pretrained(str(path), torch_dtype=dtype)
        tr = tr.eval().to(dev)
        imgs[name] = diffusers_images(mod, tr, vae, enc, n, name, out)
        del tr
        torch.cuda.empty_cache()
    del vae
    torch.cuda.empty_cache()
    nonfinite = {
        name: sum(not bool(torch.isfinite(x).all()) for x in v.values())
        for name, v in imgs.items()
    }
    return imgs, {"nonfinite_images": nonfinite}, {"embeds": "official encode_prompt"}


def card_snippet(prompts, out) -> dict:
    """The model card's diffusers usage, run as written (local paths in place of the
    repo ids): the official pipeline in fp16 with the fixed transformer swapped in."""
    try:
        pipe = DiffusionPipeline.from_pretrained(
            PIPE_ROOT,
            custom_pipeline="pipeline_moss_image",
            trust_remote_code=True,
            transformer=None,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
        cls = importlib.import_module(type(pipe).__module__).MOSSImageTransformer
        pipe.transformer = cls.from_pretrained(FIX_DIR, torch_dtype=torch.float16)
        pipe.to(DEVICE)
        pipe.set_progress_bar_config(disable=True)
        image = pipe(
            prompts[0],
            height=SIZE[0],
            width=SIZE[1],
            num_inference_steps=STEPS,
            guidance_scale=CFG,
            generator=torch.Generator(DEVICE).manual_seed(SEEDS[0]),
            output_type="pt",
        ).images
        px = (image.float().cpu() * 2 - 1).clamp(-1, 1)
        save_png(px, out / "images" / "diffusers" / f"snippet_p0_s{SEEDS[0]}.png")
        del pipe
        torch.cuda.empty_cache()
        return {"ok": True, "finite": bool(torch.isfinite(px).all()), "px": px}
    except Exception as err:  # noqa: BLE001
        log.warning(f"card snippet failed: {err!r}")
        torch.cuda.empty_cache()
        return {"ok": False, "error": repr(err)}


def comfy_images(prompts) -> dict:
    """ComfyUI images written by the ComfyUI-venv runner: <config>_p<i>_s<seed>.png."""
    imgs = {}
    for path in sorted(Path(COMFY_DIR).glob("*_p*_s*.png")):
        name, key = path.stem.split("_p", 1)
        imgs.setdefault(name, {})[f"p{key}"] = load_png(path)
    return imgs


def main() -> None:
    setup()
    check_script(globals())
    if not ROOT or not STUDY or not OUT or not FIX_FILE or not FIX_DIR:
        raise SystemExit("set ROOT, STUDY, FIX_FILE, FIX_DIR and OUT")
    apply_tune("off")
    root, study, out = Path(ROOT), Path(STUDY), Path(OUT)
    te_root = Path(TE_VAE_ROOT or ROOT)
    (out / "latents").mkdir(parents=True, exist_ok=True)
    prompts = json.loads((study / "report.json").read_text())["prompts"]
    metric = lpips.LPIPS(net="alex", verbose=False).to(DEVICE)
    report = {"prompts": prompts, "seeds": list(SEEDS), "size": list(SIZE)}
    report.update(steps=STEPS, cfg=CFG, fix_file=FIX_FILE, fix_dir=FIX_DIR)
    report["files"] = file_checks(root)
    log.info(json.dumps(report["files"], default=str))
    runners = {"kohakumoss": kohakumoss_runtime, "diffusers": diffusers_runtime}
    for rt in RUNTIMES:
        args = (
            (root, te_root, study, out)
            if rt == "kohakumoss"
            else (root, te_root, prompts, out)
        )
        imgs, checks, info = runners[rt](*args)
        others = [n for n in imgs if n != "fp32"]
        report[rt] = {
            "vs_fp32": score(metric, imgs, "fp32", others),
            "fix_vs_bf16": score(metric, imgs, "bf16", ["fix_fp16"])["fix_fp16"],
            "lora": lora_scores(metric, imgs),
            "checks": checks,
            **info,
        }
        if rt == "diffusers" and PIPE_ROOT:
            snip = card_snippet(prompts, out)
            if snip.get("ok"):
                ref = imgs["fix_fp16"][f"p0_s{SEEDS[0]}"]
                with torch.inference_mode():
                    d = float(metric(snip["px"].to(DEVICE), ref.to(DEVICE)).item())
                snip = {"ok": True, "finite": snip["finite"], "lpips_vs_fix_fp16": d}
            report[rt]["card_snippet"] = snip
        del imgs
        (out / "quality.json").write_text(json.dumps(report, indent=1, default=str))
        log.info(
            json.dumps(
                {rt: {k: v for k, v in report[rt].items() if k != "vs_fp32"}},
                default=str,
            )
        )
    if COMFY_DIR:
        imgs = comfy_images(prompts)
        others = [n for n in imgs if n != "fp32"]
        report["comfyui"] = {
            "vs_fp32": score(metric, imgs, "fp32", others) if "fp32" in imgs else {},
            "fix_vs_bf16": score(metric, imgs, "bf16", ["fix_fp16"])["fix_fp16"],
            "lora": lora_scores(metric, imgs),
            "run": json.loads((Path(COMFY_DIR) / "comfy.json").read_text()),
        }
    (out / "quality.json").write_text(json.dumps(report, indent=1, default=str))
    log.info(f"wrote {out / 'quality.json'}")


if __name__ == "__main__":
    main()
