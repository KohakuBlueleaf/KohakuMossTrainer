"""Training on a CUDA device: fp16 scaler, bf16 autocast, fused AdamW, Muon, entry runs."""

import pytest
import torch

from kohakumoss.diffusion.objective import MSELoss, TokenNormalizer, no_weighting
from kohakumoss.diffusion.timestep import MossTimestep
from kohakumoss.model import random_backbone
from kohakumoss.training.entry import run
from kohakumoss.training.flops import peak_tflops
from kohakumoss.training.full import FullTrainable
from kohakumoss.training.optim import build_bundle
from kohakumoss.training.precision import autocast_factory, build_policy
from kohakumoss.training.step import DeviceBatcher, TrainStep
from kohakumoss_kernels import ops as kernel_ops
from kohakumoss_kernels.hardware.arch import arch_of
from tests.training._train_fakes import FakeLoader
from tests.training.test_entry_smoke import smoke_knobs

pytestmark = pytest.mark.cuda


def _train(precision: str, optimizer: str, steps: int = 8, inject_inf_at: int = -1):
    dev = torch.device("cuda")
    policy = build_policy(precision, ("time_embedding",))
    model = random_backbone("tiny", device=dev, dtype=policy.compute, seed=0)
    trainable = FullTrainable(model)
    for p in trainable.trainable_params():
        p.data = p.data.float()
    groups = trainable.param_groups(
        1e-3, weight_decay=0.01, rules="moss", use_muon=optimizer == "muon"
    )
    bundle = build_bundle(
        groups,
        dev,
        optimizer,
        {"lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8},
        {"mode": "constant", "end": -1},
        steps,
        policy,
        1.0,
    )
    step = TrainStep(
        model,
        MSELoss(model.ops.token_mse),
        no_weighting,
        MossTimestep(),
        TokenNormalizer(),
        autocast_factory(policy, dev),
        dev,
        seed=0,
    )
    loader = iter(FakeLoader(model.cfg.latent_channels, model.cfg.text_dim))
    batcher = DeviceBatcher(dev)
    oks, losses = [], []
    for i in range(steps):
        batch = batcher(next(loader))
        out = step(batch, step.denominator(batch), step=i)
        loss = out.loss * (float("inf") if i == inject_inf_at else 1.0)
        bundle.backward(loss)
        oks.append(bundle.step())
        losses.append(bundle.last_loss)
    return oks, losses, bundle


def _skip_no_bf16(precision: str) -> None:
    if (
        precision == "bf16"
        and torch.cuda.is_available()
        and not arch_of("cuda").has_bf16
    ):
        pytest.skip("arch has no bf16")


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_mixed_precision_steps(precision):
    _skip_no_bf16(precision)
    oks, losses, bundle = _train(precision, "adamw")
    assert all(oks) and all(v == v and v < 1e3 for v in losses)


def test_fp16_overflow_skips_and_recovers():
    oks, losses, bundle = _train("fp16", "adamw", steps=6, inject_inf_at=2)
    assert oks == [True, True, False, True, True, True]
    assert bundle.scaler.scale == 2.0**15 and bundle.skipped == 1


def test_muon_on_cuda():
    oks, losses, _ = _train("fp16", "muon")
    assert all(oks) and losses[-1] == losses[-1]


def test_peak_known():
    assert peak_tflops(torch.device("cuda")) is not None


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_entry_lora_on_cuda(tmp_path, precision):
    _skip_no_bf16(precision)
    eager = {
        op: "eager" for op in kernel_ops.NORM + kernel_ops.ROPE + kernel_ops.ATTENTION
    }
    eager.update(
        {op: "eager" for op in kernel_ops.FFN + kernel_ops.RESIDUAL + kernel_ops.LOSS}
    )
    knobs = smoke_knobs(
        tmp_path, PRECISION=precision, STEPS=4, SAVE_EVERY=0, KERNELS=eager
    )
    state = run(knobs)
    assert state.step == 4 and state.skipped == 0
