"""Shared sampling defaults for scripts/sample/sample.py (the official sampler settings:
50 steps, CFG 3, shift 5) on a ComfyUI models folder (an HF pipeline folder works the
same as WEIGHTS); layers and starter configs override from here.

    kogine run scripts/sample/sample.py --config configs/sample/_base/base_sample.py
"""

WEIGHTS = "ComfyUI/models"
PROMPT = "a lighthouse on a rocky coast at sunset"
SIZES = [(1024, 1024)]
STEPS = 50
CFG = 3.0
SHIFT = 5.0
OUT = "runs/samples"
