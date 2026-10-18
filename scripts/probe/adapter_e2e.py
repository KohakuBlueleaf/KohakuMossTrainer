"""Real-weight adapter round trip: a random LoRA on the live model vs the same LoRA
exported (comfy / lycoris) and loaded by MossPipeline (merge / bypass), fp32.

    kogine run scripts/probe/adapter_e2e.py --set WEIGHTS=<root> --set EMBEDS=embeds.pt \
        --set OUT=runs/e2e
"""

import gc
import json
from pathlib import Path

import torch

from kohakumoss.adapters import AdapterConfig, apply
from kohakumoss.inference import MossPipeline
from kohakumoss.logging.log import get_logger, setup
from kohakumoss.model import load_backbone
from kohakumoss.probe.official import rel_error
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_common.knobs import check_script

log = get_logger("scripts.adapter_e2e")

WEIGHTS = ""  # weights root containing base/
EMBEDS = ""  # embeddings file from scripts/probe/embed.py
OUT = ""  # output directory (exported files, adapter_e2e.json)
RANK = 8  # LoRA rank
MULT = 0.8


def free():
    gc.collect()
    torch.cuda.empty_cache()


def probe_inputs(dev, text):
    g = torch.Generator().manual_seed(3)
    img = torch.randn((32 * 32, 128), generator=g).to(dev)
    seq = build_seqinfo(torch.tensor([[32, 32]]), torch.tensor([text.shape[0]]))
    return img, text.to(dev), torch.tensor([0.6], device=dev), seq.to(dev)


def run(model, dev, data):
    with torch.inference_mode():
        v = model(*probe_inputs(dev, data["embeds"][0])).float().cpu()
    res = MossPipeline.from_components(model, device=dev).generate(
        prompt_embeds=[data["embeds"][0].to(dev)],
        negative_embeds=[data["empty"].to(dev)],
        sizes=(512, 512),
        steps=20,
        seeds=[7],
        output="latent",
    )
    return v, res.latents[0].float().cpu()


def main() -> None:
    setup()
    check_script(globals())
    if not WEIGHTS or not EMBEDS or not OUT:
        raise SystemExit("set WEIGHTS, EMBEDS and OUT")
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    data = torch.load(EMBEDS)
    base_dir = Path(WEIGHTS) / "base"
    live, _ = load_backbone(base_dir, device=dev, dtype=torch.float32)
    live.requires_grad_(False)
    v_base, _ = run(live, dev, data)
    config = AdapterConfig(algo="lora", preset="attn-mlp", dim=RANK, alpha=RANK)
    aset = apply(live, config)
    torch.manual_seed(11)
    with torch.no_grad():
        for name, p in aset.named_trainable():
            if "lora_up" in name:
                p.normal_(0, 0.002)
    aset.set_multiplier(MULT)
    live.bind_ops()
    v_live, lat_live = run(live, dev, data)
    for fmt in ("comfy", "lycoris"):
        aset.export(str(out / f"{fmt}.safetensors"), fmt, dtype=None)
    aset.export(str(out / "comfy_bf16.safetensors"), "comfy")
    del live, aset
    free()
    report = {"adapter_effect_forward": rel_error(v_live, v_base), "runs": {}}
    for fmt in ("comfy", "lycoris", "comfy_bf16"):
        for mode in ("merge", "bypass"):
            pipe = MossPipeline.from_pretrained(
                WEIGHTS,
                variant="base",
                dtype=torch.float32,
                device=dev,
                adapters=[(str(out / f"{fmt}.safetensors"), MULT)],
                adapter_mode=mode,
                text_encoder=False,
                vae=False,
            )
            v, lat = run(pipe.backbone, dev, data)
            row = {
                "forward_rel": rel_error(v, v_live),
                "latent_rel_20_steps": rel_error(lat, lat_live),
            }
            report["runs"][f"{fmt}/{mode}"] = row
            log.info(f"{fmt} {mode} {row}")
            del pipe
            free()
    (out / "adapter_e2e.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    log.info(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
