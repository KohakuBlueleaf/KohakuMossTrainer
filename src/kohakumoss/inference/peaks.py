"""Estimated and measured device memory of a generate call: the DiT inference estimate
of its memory plan at the call's largest packed forward, the text encoder's placement
sizes and the VAE's weights, against the accelerator's measured peak."""

import torch

from kohakumoss.memory.estimate import GIB, Workload, estimate_inference
from kohakumoss.offload.encoders.vae import vae_act_bytes
from kohakumoss.offload.weights.layers import module_bytes
from kohakumoss.types.plans import MemoryPlan
from kohakumoss_kernels.hardware.memory import device_used


def dit_estimate(backbone, plan: MemoryPlan, tokens: int):
    """estimate_inference of `backbone` under `plan` for one `tokens`-token forward."""
    cb = next(backbone.parameters()).element_size()
    work = Workload(
        arch=backbone.cfg, img_tokens=tokens, txt_tokens=0, compute_bytes=cb
    )
    return estimate_inference(work, plan)


def working_set(backbone, plan: MemoryPlan, tokens: int) -> int:
    """Bytes one `tokens`-token DiT forward adds on top of its placed weights."""
    est = dit_estimate(backbone, plan, tokens)
    return max(est.total - est.parts.get("weights", 0), 0)


def estimated(backbone, plan: MemoryPlan, tokens: int, place, pixels: int):
    """Per phase of a generate call, GiB on the device: "encode" (the DiT's placed
    weights + the text encoder's pass + a resident VAE), "sampling" (the DiT forward of
    `tokens` + what stays resident between passes), "decode" (the DiT weights + a
    resident text encoder + one VAE pass over `pixels`), "peak" the largest; plus the
    parts "dit", "te_resident", "te_pass", "vae". `place` holds the pipeline's text
    encoder and VAE."""
    est = dit_estimate(backbone, plan, tokens)
    weights = est.parts.get("weights", 0)
    te, vae = place.text_encoder, place.vae
    fp = te.footprint() if te is not None and hasattr(te, "footprint") else {}
    te_res, te_pass = fp.get("resident_bytes", 0), fp.get("pass_bytes", 0)
    vae_w = module_bytes(vae.vae) if vae is not None else 0
    vae_res = vae_w if place.vae_on_device else 0
    vae_pass = vae_w + vae_act_bytes(pixels, vae.plan.tile) if vae is not None else 0
    out = {"dit": est.total, "te_resident": te_res, "te_pass": te_pass, "vae": vae_res}
    out["encode"] = weights + te_pass + vae_res
    out["sampling"] = est.total + te_res + vae_res
    out["decode"] = weights + te_res + vae_pass
    out["peak"] = max(out["encode"], out["sampling"], out["decode"])
    return {k: round(v / GIB, 2) for k, v in out.items()}


def accelerator(device: torch.device):
    """The torch.cuda / torch.xpu module of `device`, None on the host."""
    if device.type not in ("cuda", "xpu"):
        return None
    mod = torch.get_device_module(device)
    return mod if mod.is_available() else None


def vram_summary(device: torch.device, alloc: int, held: int) -> dict:
    """A peak window's bytes as the labelled GiB record of reports and metadata:
    `peak_alloc_gib` (live tensors), `peak_reserved_gib` (the torch allocator's hold),
    `device_used_gib` (the whole device now, every process: nvidia-smi's number);
    empty on the host."""
    if accelerator(device) is None:
        return {}
    return {
        "peak_alloc_gib": round(alloc / GIB, 3),
        "peak_reserved_gib": round(held / GIB, 3),
        "device_used_gib": round(device_used(device) / GIB, 3),
    }
