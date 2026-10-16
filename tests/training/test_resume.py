"""Native driver on the tiny MOSS: bit-exact resume and checkpoint layout."""

import pytest
import torch

from kohakumoss.diffusion.objective import MSELoss, TokenNormalizer, no_weighting
from kohakumoss.diffusion.timestep import MossTimestep
from kohakumoss.model import random_backbone
from kohakumoss.training.callbacks import CallbackRunner, SaveCheckpoint
from kohakumoss.training.checkpoint import CheckpointManager
from kohakumoss.training.drivers import NativeDriver, Parts
from kohakumoss.training.ema import ConstantEMA
from kohakumoss.training.full import FullTrainable
from kohakumoss.training.optim import build_bundle
from kohakumoss.training.precision import autocast_factory, build_policy
from kohakumoss.training.state import TrainState
from kohakumoss.training.step import DeviceBatcher, TrainStep
from tests._util.negative import must_fail
from tests.training._train_fakes import FakeLoader

pytestmark = pytest.mark.cpu
DEV = torch.device("cpu")


def make_run(out_dir, steps, restore_rng=True):
    model = random_backbone("tiny", device=DEV, dtype=torch.float32, seed=0)
    trainable = FullTrainable(model)
    policy = build_policy("fp32", ())
    groups = trainable.param_groups(1e-3, weight_decay=0.01, rules="moss")
    bundle = build_bundle(
        groups,
        DEV,
        "adamw",
        {"lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8},
        {"mode": "cosine", "warmup": 2, "end": -1},
        6,
        policy,
        1.0,
    )
    normalizer = TokenNormalizer()
    step = TrainStep(
        model,
        MSELoss(model.ops.token_mse),
        no_weighting,
        MossTimestep(),
        normalizer,
        autocast_factory(policy, DEV),
        DEV,
        seed=3,
    )
    ema = ConstantEMA(trainable.named_trainable(), (0.9,))
    loader = FakeLoader(model.cfg.latent_channels, model.cfg.text_dim)
    ckpt = CheckpointManager(out_dir, keep_last=2)
    parts = Parts(trainable, ema, bundle, step, loader)
    runner = CallbackRunner([SaveCheckpoint(3, ckpt, parts)])
    driver = NativeDriver(
        steps, 2, DeviceBatcher(DEV), normalizer, DEV, runner, ckpt, parts
    )
    return model, driver, step, loader, bundle, ema, ckpt, parts


def _params(model):
    return [p.detach().clone() for p in model.parameters()]


def resume_matches(tmp_path, restore_rng: bool):
    model_a, drv, step, loader, bundle, ema, _, _ = make_run(tmp_path / "a", 6)
    drv.run(step, loader, bundle, ema, TrainState())
    model_b, drv, step, loader, bundle, ema, _, _ = make_run(tmp_path / "b", 3)
    drv.run(step, loader, bundle, ema, TrainState())
    model_c, drv, step, loader, bundle, ema, ckpt, parts = make_run(tmp_path / "b", 6)
    state = TrainState()
    loaded = ckpt.load(ckpt.latest())
    if not restore_rng:
        loaded["step_rng"] = step.rng_state()
    parts.restore(loaded, state)
    assert state.step == 3
    drv.run(step, loader, bundle, ema, state)
    for p, q in zip(_params(model_a), _params(model_c)):
        assert torch.equal(p, q)


def test_resume_is_bit_exact(tmp_path):
    resume_matches(tmp_path, restore_rng=True)
    files = {
        p.name for p in (tmp_path / "b" / "checkpoints" / "step_00000006").iterdir()
    }
    assert {
        "trainable.safetensors",
        "ema.safetensors",
        "optim.pt",
        "rng.pt",
        "state.json",
    } <= files


def test_resume_without_step_rng_diverges(tmp_path):
    must_fail(
        lambda: resume_matches(tmp_path, restore_rng=False), "step RNG not restored"
    )


def test_loss_decreases_on_fixed_batch(tmp_path):
    model, drv, step, loader, bundle, ema, _, _ = make_run(tmp_path / "c", 1)
    batch = DeviceBatcher(DEV)(next(iter(loader)))
    losses = []
    denom = step.denominator(batch)
    for i in range(30):
        out = step(batch, denom, step=i)
        bundle.backward(out.loss)
        bundle.step()
        losses.append(bundle.last_loss)
    assert losses[-1] < losses[0]
