"""Knob checks of exports, sampling / validation inputs, checkpoints and paths."""

import os
from pathlib import Path

EXPORTS = ("comfy", "lycoris", "merged", "diffusers")


def io_problems(k: dict) -> list[str]:
    """EXPORT_FORMATS, SAMPLE / VALIDATION inputs, checkpoint cadence, RESUME and
    WEIGHTS paths."""
    p = []
    bad = [f for f in k["EXPORT_FORMATS"] if f not in EXPORTS]
    if bad:
        p.append(f"EXPORT_FORMATS has unknown formats {bad}; valid {EXPORTS}")
    if k["ALGO"] == "full" and set(k["EXPORT_FORMATS"]) & {"comfy", "lycoris"}:
        p.append("full finetune exports 'merged' / 'diffusers', not adapter formats")
    if k["SAMPLE_EVERY"] > 0 and not k["SAMPLE_PROMPTS"]:
        p.append("SAMPLE_EVERY > 0 needs SAMPLE_PROMPTS")
    if k["VALIDATE_EVERY"] > 0 and not k["VALIDATION"].get("datasets"):
        p.append("VALIDATE_EVERY > 0 needs VALIDATION['datasets']")
    if k["SAVE_EVERY"] < 0 or k["KEEP_LAST"] < 0:
        p.append("SAVE_EVERY and KEEP_LAST must be >= 0")
    resume = k["RESUME"]
    if resume not in ("", "latest") and not Path(resume).exists():
        p.append(f"RESUME={resume!r} does not exist")
    weights = k["WEIGHTS"]
    if weights and not os.path.exists(weights):
        p.append(f"WEIGHTS={weights!r} does not exist")
    return p
