"""fp8 / MXFP8 inference on real weights, every precision in one report.

Stages (one JSON report, `fp8_inference.json`; the text stage always runs):
  text     : text-encoder embeddings of each precision against the host fp64 encode
             of the bf16 weights next to bf16; encode time, weight and peak VRAM
  dit      : DiT velocity of each base precision at SIZES[0] against the host fp64
             forward of the bf16 weights next to bf16; weight VRAM, forward time
  generate : (with dit, same load) STEPS-step generate of BATCH prompts (time, peak
             VRAM), final latents and decoded images (PNGs) against bf16

    kogine run scripts/probe/fp8_inference.py --set WEIGHTS=<root> \
        --set OUT=runs/probe/fp8
"""

import gc
import json
import time
from functools import partial
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from kohakumoss.encoders import FluxVae, QwenTextEncoder
from kohakumoss.encoders.text.tokens import tokenizer_dir
from kohakumoss.inference import MossPipeline
from kohakumoss.inference.loading import component_root
from kohakumoss.inference.pipeline import to_pil
from kohakumoss.logging.log import get_logger, log_object, setup
from kohakumoss.model import load_backbone
from kohakumoss.model.opset import OpSet
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_common.knobs import check_script
from kohakumoss_kernels.error_stats import compare, stats
from kohakumoss_kernels.hardware.memory import device_used, window

log = get_logger("scripts.fp8_inference")

WEIGHTS = ""  # weights root (base/, sft/) or one pipeline dir
VARIANT = "base"
OUT = ""  # output directory
DEVICE = "cuda"
PRECISIONS = ["compute", "fp8_compute:row", "fp8_compute:tensor", "fp8_compute:block", "mxfp8"]  # fmt: skip
STAGES = ("text", "dit", "generate")
REF64 = True  # host fp64 references for the text and dit stages
TUNE = "table"  # kernel tuning of the DiT pipeline: "table" | "off"
PROMPTS = [
    "a red fox sitting in a snowy forest at dawn, soft light",
    "an isometric illustration of a tiny city on a floating island",
    "portrait photo of an old fisherman, dramatic lighting, 85mm",
    "a bowl of ramen on a wooden table, top view",
]
SIZES = [(1024, 1024)]  # (H, W) per generated prompt, the last one repeating
SIGMA = 0.7
STEPS = 20
BATCH = 2
CFG = 3.0
REPEATS = 3


def free():
    gc.collect()
    torch.cuda.empty_cache()


def split(spec: str) -> tuple[str, str]:
    precision, _, recipe = spec.partition(":")
    return precision, recipe or "row"


def timed_peak(fn, dev):
    """(result, best seconds of REPEATS, peak allocated bytes above the allocation
    before, peak reserved bytes, device-used bytes after)."""
    fn()
    torch.cuda.synchronize(dev)
    base = torch.cuda.memory_allocated(dev)
    best, out = float("inf"), None
    with window(dev) as w:
        for _ in range(REPEATS):
            start = time.perf_counter()
            out = fn()
            torch.cuda.synchronize(dev)
            best = min(best, time.perf_counter() - start)
    return out, best, w.alloc - base, w.reserved, device_used(dev)


def table(out, bf16, ref64) -> dict:
    t = compare(out.float().cpu(), bf16.float().cpu(), ref64)
    t["vs_bf16"] = stats(out.float().cpu(), bf16.double().cpu())
    return t


def stage_text(root: Path, report: dict) -> dict:
    dev = torch.device(DEVICE)
    rows, embeds = {}, {}
    for spec in PRECISIONS:
        precision, recipe = split(spec)
        plan = {"memory": "resident", "dtype": "bf16", "precision": precision}
        plan["fp8_recipe"] = recipe
        te = QwenTextEncoder.from_pretrained(root, dev, plan, output_device="cpu")
        weights = torch.cuda.memory_allocated(dev)
        encode = partial(te.encode, [*PROMPTS, ""])
        out, sec, peak, held, used = timed_peak(encode, dev)
        embeds[spec] = out
        rows[spec] = {
            "encode_s": sec,
            "weights_gib": weights / 2**30,
            "pass_peak_alloc_gib": peak / 2**30,
            "pass_peak_reserved_gib": held / 2**30,
            "device_used_gib": used / 2**30,
            "quant": te.quant,
        }
        log_object(log, "text encoder", {"spec": spec, **rows[spec]})
        del te
        free()
    if REF64:
        model_dir = root / "text_encoder"
        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir(model_dir)))
        model = AutoModel.from_pretrained(str(model_dir), dtype=torch.bfloat16).double()
        host = QwenTextEncoder(model, tok, "cpu", {"memory": "cpu"})
        start = time.perf_counter()
        ref = host.encode([*PROMPTS, ""])
        log.info(f"text fp64 reference in {time.perf_counter() - start:.0f} s")
        del host, model
        gc.collect()
        ref64 = torch.cat(ref).double()
        bf16 = torch.cat(embeds["compute"])
        for spec in PRECISIONS:
            rows[spec]["precision"] = table(torch.cat(embeds[spec]), bf16, ref64)
    report["text"] = rows
    return embeds


def velocity(model, latent, text):
    _, c, h, w = latent.shape
    dev = next(model.parameters()).device
    seq = build_seqinfo(torch.tensor([[h, w]]), torch.tensor([text.shape[0]])).to(dev)
    img = latent[0].flatten(1).T.to(dev)
    sigma = torch.tensor([SIGMA], device=dev)
    with torch.inference_mode():
        return model(img, text.to(dev), sigma, seq).float().cpu()


def _memory(spec: str) -> tuple[str | None, dict]:
    precision, recipe = split(spec)
    if precision == "compute":
        return None, {}
    return "speed", {"base_precision": precision, "fp8_recipe": recipe}


def load_pipe(spec: str) -> MossPipeline:
    memory, overrides = _memory(spec)
    return MossPipeline.from_pretrained(
        WEIGHTS,
        variant=VARIANT or None,
        dtype=torch.bfloat16,
        device=DEVICE,
        text_encoder=False,
        vae=False,
        memory=memory,
        memory_overrides=overrides,
        tune=TUNE,
    )


def generate(pipe, embeds: dict, dev) -> tuple[list, dict]:
    """Final latents and time / peak of a STEPS-step generate of BATCH prompts (after a
    2-step warm call that tunes the kernels)."""
    pos = [e.to(dev, torch.bfloat16) for e in embeds["compute"][:BATCH]]
    neg = [embeds["compute"][-1].to(dev, torch.bfloat16)] * len(pos)
    kw = dict(prompt_embeds=pos, negative_embeds=neg, sizes=list(SIZES), cfg=CFG)
    kw.update(seeds=list(range(len(pos))), batch=BATCH, output="latent")
    pipe.generate(steps=2, **kw)
    torch.cuda.synchronize(dev)
    start = time.perf_counter()
    res = pipe.generate(steps=STEPS, **kw)
    torch.cuda.synchronize(dev)
    sec = time.perf_counter() - start
    row = {"generate_s": sec, "s_per_step": sec / STEPS}
    row.update({f"generate_{k}": v for k, v in res.settings["vram"].items()})
    return [z.float().cpu() for z in res.latents], row


def stage_dit(root: Path, embeds: dict, report: dict) -> None:
    dev = torch.device(DEVICE)
    text = embeds["compute"][0].to(torch.bfloat16)
    g = torch.Generator().manual_seed(7)
    latent = torch.randn((1, 128, SIZES[0][0] // 16, SIZES[0][1] // 16), generator=g)
    rows, outs, lat = {}, {}, {}
    for spec in PRECISIONS:
        pipe = load_pipe(spec)
        weights = torch.cuda.memory_allocated(dev)
        fwd = partial(velocity, pipe.backbone, latent.to(torch.bfloat16), text)
        outs[spec], sec, peak, held, used = timed_peak(fwd, dev)
        rows[spec] = {"weights_gib": weights / 2**30, "forward_s": sec}
        rows[spec]["forward_peak_alloc_gib"] = peak / 2**30
        rows[spec]["forward_peak_reserved_gib"] = held / 2**30
        rows[spec]["device_used_gib"] = used / 2**30
        if "generate" in STAGES:
            lat[spec], gen = generate(pipe, embeds, dev)
            rows[spec].update(gen)
        log_object(log, "dit", {"spec": spec, **rows[spec]})
        del pipe, fwd
        free()
    report["dit"] = rows
    if lat:
        compare_images(root, lat, rows)
    if REF64:
        model, _ = load_backbone(root, device="cpu", dtype=torch.bfloat16)
        model = model.double()
        model.bind_ops(OpSet.resolve("cpu", torch.float64))
        start = time.perf_counter()
        ref64 = velocity(model, latent.double(), text.double()).double()
        log.info(f"dit fp64 reference in {time.perf_counter() - start:.0f} s")
        del model
        gc.collect()
        for spec in PRECISIONS:
            rows[spec]["precision"] = table(outs[spec], outs["compute"], ref64)


def _flat(zs: list) -> torch.Tensor:
    return torch.cat([z.flatten() for z in zs])


def compare_images(root: Path, lat: dict, rows: dict) -> None:
    """Decode every precision's latents (PNGs in OUT); latent / pixel stats vs bf16."""
    dev = torch.device(DEVICE)
    out_dir = Path(OUT)
    vae = FluxVae.from_pretrained(root, dev, torch.float32)
    images = {}
    with torch.inference_mode():
        for spec, zs in lat.items():
            images[spec] = []
            for i, z in enumerate(zs):
                img = vae.decode(z.to(dev)).cpu()
                images[spec].append(img)
                to_pil(img).save(out_dir / f"{spec.replace(':', '_')}_{i}.png")
    del vae
    free()
    for spec in PRECISIONS:
        base_lat, base_img = _flat(lat["compute"]).double(), _flat(images["compute"])
        rows[spec]["latent_vs_bf16"] = stats(_flat(lat[spec]), base_lat)
        rows[spec]["image_vs_bf16"] = stats(_flat(images[spec]), base_img.double())


def main() -> None:
    setup()
    check_script(globals())
    if not WEIGHTS or not OUT:
        raise SystemExit("set WEIGHTS and OUT")
    if "compute" not in PRECISIONS:
        raise SystemExit("PRECISIONS must include 'compute' (the bf16 baseline)")
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    root = component_root(Path(WEIGHTS), VARIANT or None, "transformer")
    report = {"precisions": PRECISIONS, "sizes": SIZES, "steps": STEPS, "batch": BATCH}
    path = out / "fp8_inference.json"
    embeds = stage_text(root, report)
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if "dit" in STAGES:
        stage_dit(root, embeds, report)
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    log.info(f"wrote {path}")


if __name__ == "__main__":
    main()
