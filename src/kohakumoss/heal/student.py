"""Teacher and student of a heal run, built by the framework from the training knobs:
the teacher frozen and placed by its memory plan, the student with LoKr adapters and
TRAIN_NORMS gain deltas, shrunk by the calibration report and bound to weighted norms
when the run anneals."""

import json
import re
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

import torch
from torch.nn.utils import parametrize

from kohakumoss.cli.config.validate import load_knobs
from kohakumoss.heal.calibrate import report_scales
from kohakumoss.heal.channels import scaled, shrink_table
from kohakumoss.heal.config import NORM_TARGETS, WRITERS_FULL, HealConfig
from kohakumoss.heal.sites import model_sites, norm_path
from kohakumoss.heal.wnorm import GainAnneal, WeightedNorms, anneal_lam
from kohakumoss.memory.apply import apply_to_backbone, quantized_weights
from kohakumoss.model.opset import OpSet
from kohakumoss.precision import autocast_factory, build_precision
from kohakumoss.training.build.model import build_backbone, build_model, resolve_memory
from kohakumoss.training.knobs import check

FIXED = {
    "ALGO": "lokr",
    "TRAIN_NORMS": True,
    "PRECISION": "bf16",
    "FP16_FIX": "none",
    "FP16_FIX_KWARGS": {},
    "BASE_PRECISION": "compute",
    "INT8_COMPUTE": False,
    "TEXT_CACHE": "offline",
    "LATENT_CACHE": "offline",
    "SAMPLE_EVERY": 0,
    "ENCODE_DEVICES": (),
    "DEVICES": (0,),
    "TEXT_DEVICE": "",
    "PARALLEL": "none",
    "MEMORY_PROFILE": "manual",
    "CHECKPOINT": "block",
    "TUNE": "off",
}


def heal_knobs(
    train_script: Path, train_config: str, c: HealConfig
) -> tuple[dict, dict]:
    """(student knobs, teacher knobs): the training script's defaults, the training
    config (model, kernels, memory knobs), the heal's fixed choices, then each model's
    memory overrides."""
    base = load_knobs(train_script)
    if train_config:
        base |= load_knobs(Path(train_config))
    preset = {"target_name": [c.lokr_regex], "norm_name": list(NORM_TARGETS)}
    base |= FIXED | {
        "PRESET": preset,
        "ALGO_KWARGS": {"factor": c.lokr_factor, "full_matrix": True},
    }
    student = check(base | {"TRANSFORMER": c.student_file} | c.student_memory)
    teacher = check(base | {"TRANSFORMER": c.teacher} | c.teacher_memory)
    return student, teacher


def build_frozen(k: dict, device: torch.device, prec):
    """A frozen DiT placed by the knobs' memory plan, no adapters: (backbone, engines)."""
    plan, _ = resolve_memory(k, device, prec.compute, 1)
    defer = partial(
        quantized_weights, plan=plan, trainable_base=False, keep_fp32=prec.keep_fp32
    )
    backbone = build_backbone(k, torch.device("cpu"), prec, defer)
    engines = apply_to_backbone(
        backbone,
        plan,
        device,
        prec.compute,
        training=True,
        trainable_base=False,
        keep_fp32=prec.keep_fp32,
    )
    backbone.requires_grad_(False)
    return backbone, engines


def stored_gain(model, path: str) -> torch.Tensor:
    """The stored gain of norm `path` (the parametrized original under TRAIN_NORMS)."""
    mod = model.get_submodule(path)
    if parametrize.is_parametrized(mod, "weight"):
        return mod.parametrizations.weight.original
    return mod.weight


@dataclass
class Models:
    """The built pair and what the anneal moves."""

    k: dict
    built: object
    student: torch.nn.Module
    teacher: torch.nn.Module
    teacher_engines: object
    autocast: object
    d: torch.Tensor
    wn: WeightedNorms | None
    gains: GainAnneal = field(default_factory=GainAnneal)

    @property
    def massive(self) -> torch.Tensor:
        return (self.d < 1).nonzero().flatten()

    @torch.no_grad()
    def set_anneal(self, t: float) -> None:
        """Norm channel weights (1 / D^2)^(1 - t) and gains g0 * r^-t."""
        if self.wn is not None:
            self.wn.lam.copy_(anneal_lam(self.d, t))
        self.gains.set(t)


@torch.no_grad()
def shrink_student(m: Models, c: HealConfig) -> None:
    """Writers times D, gains times 1 / D (the weighted norm at t = 0 is then exact);
    each gain's calibrated ratio joins the gain anneal (r = 1 without ANNEAL_RBAR)."""
    report = json.loads(Path(c.report).read_text(encoding="utf-8"))
    s = m.student
    m.d, rbar = report_scales(report, c.tau, s.cfg.dim)
    m.d = m.d.to(next(s.parameters()).device)
    sites = model_sites(s)
    table = shrink_table(len(s.text_embedder.blocks), s.cfg.depth, sites, m.d)
    params = dict(s.named_parameters())
    for name, mult in table.items():
        path = name.rsplit(".", 1)[0]
        t = (
            stored_gain(s, path)
            if path in {norm_path(x) for x in sites}
            else params[name]
        )
        t.copy_(scaled(t, mult))
    for site in sites:
        m.gains.add(
            stored_gain(s, norm_path(site)), rbar[site] if c.anneal_rbar else 1.0
        )


def train_writers(student) -> list[tuple[str, torch.nn.Parameter]]:
    """MODE full_writers: the joint o / ffn_down weights train in full."""
    writer = re.compile(WRITERS_FULL)
    out = []
    for name, p in student.named_parameters():
        if writer.match(name):
            p.requires_grad_(True)
            out.append((name, p))
    return out


def build_models(c: HealConfig, k: dict, k_t: dict, device: torch.device) -> Models:
    """Student (adapters, TRAIN_NORMS), teacher (frozen), both on composed ops (no fused
    linears); the student shrunk and on weighted norms when REPORT is set."""
    prec = build_precision(
        k["PRECISION"], k["KEEP_FP32"], k["FP16_FIX"], k["FP16_FIX_KWARGS"]
    )
    built = build_model(k, device, prec, 1)
    teacher, engines = build_frozen(k_t, device, prec)
    ops = replace(OpSet.resolve(device, prec.compute, k["KERNELS"]), fused={})
    teacher.bind_ops(ops)
    student = built.backbone
    student.bind_ops(ops)
    m = Models(
        k,
        built,
        student,
        teacher,
        engines,
        autocast_factory(prec, device),
        torch.ones(student.cfg.dim, device=device),
        None,
    )
    if c.report:
        shrink_student(m, c)
        if c.anneal_steps > 0:
            m.wn = WeightedNorms(student.cfg.dim, device)
            student.bind_ops(
                replace(ops, rms_norm=m.wn.rms_norm, rms_norm_mod=m.wn.rms_norm_mod)
            )
        if c.anneal_rbar:
            m.gains.check()
    m.set_anneal(0.0 if c.anneal_steps > 0 else 1.0)
    return m
