"""One recorded real-weights training step: its LoRA backward is finite.

The step's exact inputs (noised latents, target, text embeddings, sigma, layout, loss
denominator) and adapter values come from `KOHAKUMOSS_REPRO_STEP`: the
`<output>/instability/repro_step.pt` a run writes when NONFINITE_PATIENCE stops it
(`kohakumoss.training.callbacks.repro`); the run is bf16, LoRA attn-mlp rank 16,
balanced profile, default kernels.
"""

import os

import pytest
import torch

from kohakumoss.diffusion.objective import TokenNormalizer
from kohakumoss.precision import autocast_factory, build_precision
from kohakumoss.training.build.kernels import build_adapter_prewarm, build_prewarm
from kohakumoss.training.build.loop import build_step
from kohakumoss.training.build.model import build_model
from kohakumoss.training.step import Prepared
from kohakumoss.tuning.prewarm import prewarm_devices
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels.tuning.tuner import set_mode as set_tune_mode
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]

RECIPE = dict(
    VARIANT="base",
    PRECISION="bf16",
    ALGO="lora",
    PRESET="attn-mlp",
    DIM=16,
    ALPHA=16.0,
    MEMORY_PROFILE="balanced",
    RESOLUTION_MODE="resize",
    BATCHING="tokens",
    OVERLONG="drop_tokens",
    TARGET_AREA=768 * 768,
    MAX_TOKENS=8192,
    LATENT_CACHE="offline",
    TEXT_CACHE="offline",
    SAMPLE_EVERY=0,
    TUNE="table",
)


def _built(weights: str, device: torch.device):
    k = runnable(WEIGHTS=weights, **RECIPE)
    set_tune_mode(k["TUNE"])
    torch.manual_seed(k["SEED"])
    prec = build_precision(
        k["PRECISION"], k["KEEP_FP32"], k["FP16_FIX"], k["FP16_FIX_KWARGS"]
    )
    built = build_model(k, device, prec)
    warm = prewarm_devices(device)
    build_prewarm(k, built, prec, device)(warm)
    build_adapter_prewarm(k, built, prec.compute)(warm)
    autocast = autocast_factory(prec, device)
    step = build_step(k, built, TokenNormalizer(), autocast, device)
    return built, step, autocast


def _prepared(rec: dict, device: torch.device) -> Prepared:
    seq = build_seqinfo(rec["hw"], rec["text_len"], img_keep=rec["img_keep"])
    return Prepared(
        rec["z"].to(device),
        rec["text"].to(device),
        rec["sigma"].to(device),
        seq.to(device),
        rec["target"].to(device),
        rec["weight"].to(device),
        None,
    )


def _nonfinite_grads(built, step, autocast, p, denom) -> list[str]:
    for q in built.trainable.trainable_params():
        q.grad = None
    with autocast():
        pred = step.backbone(p.z, p.text, p.sigma, p.seq)
    step.loss(pred, p, denom).backward()
    return [
        name
        for name, q in built.trainable.named_trainable()
        if q.grad is None or not torch.isfinite(q.grad).all()
    ]


def test_recorded_step_has_finite_lora_grads(weights_dir):
    path = os.environ.get("KOHAKUMOSS_REPRO_STEP")
    if not weights_dir or not path:
        pytest.skip("KOHAKUMOSS_WEIGHTS / KOHAKUMOSS_REPRO_STEP unset")
    device = torch.device("cuda")
    rec = torch.load(path)
    built, step, autocast = _built(weights_dir, device)
    with torch.no_grad():
        for name, q in built.trainable.named_trainable():
            q.copy_(rec["adapters"][name].to(q))
    p = _prepared(rec, device)
    denom = torch.tensor([rec["denom"]], dtype=torch.int64, device=device)
    bad = _nonfinite_grads(built, step, autocast, p, denom)
    assert not bad, f"{len(bad)} adapter params with non-finite grads, first {bad[:3]}"

    def poisoned_target():
        q = Prepared(p.z, p.text, p.sigma, p.seq, p.target.clone(), p.weight, None)
        q.target[0, 0] = float("inf")
        assert not _nonfinite_grads(built, step, autocast, q, denom)

    must_fail(poisoned_target, "a non-finite target passed as finite grads")
