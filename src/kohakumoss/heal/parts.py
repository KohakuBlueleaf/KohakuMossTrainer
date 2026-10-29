"""A heal run's trained parts and their merge into one plain DiT file.

A parts directory holds adapter.safetensors (LyCORIS: LoKr + norm-gain deltas),
writers.safetensors (MODE full_writers: the trained o / ffn_down weights) and
state.json (the anneal t; a snapshot before t = 1 is a weight state, not a plain-norm
model). The run directory's heal.json is the HealConfig."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kohakumoss.adapters.infer.inference import load_adapters
from kohakumoss.heal.calibrate import apply_live, report_scales
from kohakumoss.heal.channels import shrink_table
from kohakumoss.heal.config import HealConfig
from kohakumoss.heal.sites import model_sites
from kohakumoss.model import load_backbone, save_single


def save_parts(heal, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    heal.m.built.trainable.export(str(out / "adapter.safetensors"), "lycoris", {})
    if heal.writers:
        writers = {
            n: p.detach().to(torch.bfloat16).cpu().contiguous() for n, p in heal.writers
        }
        save_file(writers, str(out / "writers.safetensors"))
    (out / "state.json").write_text(json.dumps({"anneal_t": heal.t}), encoding="utf-8")


@torch.no_grad()
def merge(c: HealConfig, parts: Path, out: Path, dtype=torch.bfloat16) -> dict:
    """The student file with the run's start state rebuilt (shrunk writers and gains
    w / (D r^t) at the parts' anneal t), the trained writers, then the LyCORIS file
    merged, written as a plain single-file DiT. Raises unless the parts are at t = 1."""
    state = json.loads((parts / "state.json").read_text(encoding="utf-8"))
    t = float(state["anneal_t"])
    if t < 1.0:
        raise ValueError(
            f"{parts} is at anneal t={t}: only t = 1 is a plain-norm model"
        )
    model, _ = load_backbone(c.student_file, device="cpu", dtype=dtype)
    model.requires_grad_(False)
    if c.report:
        report = json.loads(Path(c.report).read_text(encoding="utf-8"))
        d, rbar = report_scales(report, c.tau, model.cfg.dim)
        r = rbar if c.anneal_rbar else None
        apply_live(
            model,
            shrink_table(
                len(model.text_embedder.blocks),
                model.cfg.depth,
                model_sites(model),
                d,
                r,
            ),
        )
    writers = parts / "writers.safetensors"
    if writers.exists():
        params = dict(model.named_parameters())
        with safe_open(str(writers), framework="pt") as h:
            for key in h.keys():
                params[key].copy_(h.get_tensor(key).to(params[key].dtype))
    load_adapters(
        model, [(str(parts / "adapter.safetensors"), 1.0)], mode="merge", undoable=False
    )
    save_single(model, out, dtype)
    return {
        "parts": str(parts),
        "anneal_t": t,
        "out": str(out),
        "writers": writers.exists(),
    }
