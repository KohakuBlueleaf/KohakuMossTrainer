"""TrainStep wiring against an independent recomputation of the loss."""

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.diffusion.objective import MSELoss, TokenNormalizer, no_weighting
from kohakumoss.model import random_backbone
from kohakumoss.training.callbacks.evaluate import FixedSigma
from kohakumoss.training.precision import autocast_factory, build_policy
from kohakumoss.training.step import DeviceBatcher, TrainStep
from tests._util.negative import must_fail
from tests.training._train_fakes import host_batch

pytestmark = pytest.mark.cpu
DEV = torch.device("cpu")


def _setup(sigma=0.3):
    model = random_backbone("tiny", device=DEV, dtype=torch.float32, seed=0)
    policy = build_policy("fp32", ())
    step = TrainStep(
        model,
        MSELoss(model.ops.token_mse),
        no_weighting,
        FixedSigma(sigma),
        TokenNormalizer(),
        autocast_factory(policy, DEV),
        DEV,
        seed=7,
    )
    hb = host_batch(0, 0, model.cfg.latent_channels, model.cfg.text_dim)
    return model, step, DeviceBatcher(DEV)(hb)


def _manual(model, batch, sigma, noise_seed, target_fn):
    z0 = batch.latents.float()
    eps = torch.randn(z0.shape, generator=torch.Generator().manual_seed(noise_seed))
    z = (1 - sigma) * z0 + sigma * eps
    s = torch.full((batch.seq.batch,), sigma)
    pred = model(z, batch.text, s, batch.seq)
    return F.mse_loss(pred, target_fn(z0, eps))


def test_loss_equals_manual_velocity_mse():
    model, step, batch = _setup(0.3)
    out = step(batch, step.denominator(batch))
    ref = _manual(model, batch, 0.3, 8, lambda z0, eps: eps - z0)
    assert torch.allclose(out.loss, ref, rtol=1e-5)
    model2, step2, batch2 = _setup(0.3)
    out2 = step2(batch2, step2.denominator(batch2))
    must_fail(
        lambda: _close(
            out2.loss, _manual(model2, batch2, 0.3, 8, lambda z0, eps: z0 - eps)
        ),
        "target sign flipped",
    )


def _close(a, b):
    assert torch.allclose(a, b, rtol=1e-5)


def test_backward_reaches_every_parameter():
    model, step, batch = _setup(0.6)
    step(batch, step.denominator(batch)).loss.backward()
    missing = [
        n
        for n, p in model.named_parameters()
        if p.grad is None or not p.grad.abs().sum() > 0
    ]
    assert not missing, missing
