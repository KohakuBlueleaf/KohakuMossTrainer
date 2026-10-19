"""Knob checks of exports, sampling / validation inputs, checkpoints and paths."""

import os
from pathlib import Path

from kohakumoss.memory.apply import stored_base_problems
from kohakumoss.training.exports import EXPORT_FORMATS, export_problems
from kohakumoss.training.weights import KNOBS, component, stored_dit_precision
from kohakumoss.weights import COMPONENTS, DIT, TEXT, VAE, is_comfy_folder

EXPORTS = EXPORT_FORMATS


def weights_problems(k: dict) -> list[str]:
    """WEIGHTS and the component knobs: each component the run reads exists (a ComfyUI
    folder names the files it tried), and a full finetune does not train a
    pre-quantized DiT file under BASE_PRECISION auto."""
    weights = k["WEIGHTS"]
    if not weights or not os.path.exists(weights):
        return [f"WEIGHTS={weights!r} does not exist"] if weights else []
    p = []
    comfy = is_comfy_folder(weights)
    sampling = k["SAMPLE_EVERY"] > 0
    reads = {
        DIT: True,
        TEXT: k["TEXT_CACHE"] != "offline" or sampling,
        VAE: k["LATENT_CACHE"] != "offline" or sampling,
    }
    for which in COMPONENTS:
        if not reads[which]:
            continue
        try:
            path = component(k, which)
        except FileNotFoundError as err:
            p.append(f"{KNOBS[which]}: {err}")
            continue
        named = k.get(KNOBS[which]) or comfy or which == DIT
        if named and not Path(path).exists():
            p.append(f"{KNOBS[which]}: {path} does not exist")
    stored = stored_dit_precision(k)
    p += stored_base_problems(stored, k.get("BASE_PRECISION"), k["ALGO"] == "full")
    return p


def io_problems(k: dict) -> list[str]:
    """Exports, SAMPLE / VALIDATION inputs, checkpoint cadence, RESUME and the weights
    paths."""
    p = export_problems(k)
    if k["SAMPLE_EVERY"] > 0 and not k["SAMPLE_PROMPTS"]:
        p.append("SAMPLE_EVERY > 0 needs SAMPLE_PROMPTS")
    if k["VALIDATE_EVERY"] > 0 and not k["VALIDATION"].get("datasets"):
        p.append("VALIDATE_EVERY > 0 needs VALIDATION['datasets']")
    if k["SAVE_EVERY"] < 0 or k["KEEP_LAST"] < 0:
        p.append("SAVE_EVERY and KEEP_LAST must be >= 0")
    final = k.get("FINAL_CHECKPOINT", "auto")
    if final not in ("auto", True, False):
        p.append(f"FINAL_CHECKPOINT={final!r} must be 'auto', True or False")
    resume = k["RESUME"]
    if resume not in ("", "latest") and not Path(resume).exists():
        p.append(f"RESUME={resume!r} does not exist")
    return p + weights_problems(k)
