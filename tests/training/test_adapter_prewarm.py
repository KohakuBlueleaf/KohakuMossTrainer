"""LyCORIS tier-race prewarm: LyCORIS token buckets, every adapted layer run per bucket
and grad pattern, streamed layers skipped, no gradient left behind; per-group settings
of the offload AdamWs."""

import pytest
import torch
from lycoris.kernels.plans import tune

from kohakumoss.model import random_backbone
from kohakumoss.registry import import_object
from kohakumoss.training.build import integration as I
from kohakumoss.training.build.kernels import (
    GRAD_PATTERNS,
    AdapterPrewarm,
    adapter_buckets,
)
from kohakumoss.training.optim import build_bundle
from kohakumoss.training.precision import build_policy
from tests._util.negative import must_fail
from tests.training.test_knobs import runnable

pytestmark = pytest.mark.cpu
DEV = torch.device("cpu")


def _covers(buckets, batch_max: int, row: int) -> None:
    """Every row count 1..batch_max falls in a listed LyCORIS bucket, and each listed
    value is its own bucket (a representative of exactly that key)."""
    assert all(tune.bucket_tokens(b) == b for b in buckets)
    want = {tune.bucket_tokens(n) for n in range(1, batch_max + 1)}
    assert want <= set(buckets), sorted(want - set(buckets))


@pytest.mark.parametrize("batch_max", [1, 7, 1000, 4096, 9000, 16384])
def test_adapter_buckets_cover_lycoris_buckets(batch_max):
    row = tune.ROW_BUCKET
    buckets = adapter_buckets(batch_max, row)
    _covers(buckets, batch_max, row)
    assert len(buckets) == len(set(buckets))
    must_fail(lambda: _covers(buckets[:-1], batch_max, row), "top bucket missing")


def _adapted():
    model = random_backbone("tiny", device=DEV, dtype=torch.float32, seed=0)
    model.requires_grad_(False)
    k = runnable(ALGO="lora", DIM=4, ALPHA=4.0)
    spec = import_object(I.ADAPTERS_SPEC)(k)
    aset = import_object(I.ADAPTERS_APPLY)(model, spec, dtype=torch.float32, device=DEV)
    return model, aset


def test_every_adapted_layer_runs_each_bucket_and_pattern():
    model, aset = _adapted()
    skipped = aset.targets[0]
    calls: dict[int, list] = {}

    def spy(module, args, out):
        calls.setdefault(id(module), []).append(
            (args[0].shape[0], args[0].requires_grad, torch.is_grad_enabled())
        )

    for t in aset.targets:
        t.linear.register_forward_hook(spy)
    before = {n: p.detach().clone() for n, p in aset.named_trainable()}
    buckets = [1, 4, 16]
    warm = AdapterPrewarm(aset.targets, torch.float32, buckets, {id(skipped.linear)})
    warm([DEV])
    want = sorted((n, g, e) for n in buckets for g, e in GRAD_PATTERNS)
    assert id(skipped.linear) not in calls
    for t in aset.targets[1:]:
        assert sorted(calls[id(t.linear)]) == want, t.lora_name
    for name, p in aset.named_trainable():
        assert p.grad is None and torch.equal(p, before[name]), name
    other = AdapterPrewarm(aset.targets, torch.float32, buckets, set())
    calls.clear()
    other([torch.device("meta")])
    assert calls == {}, "layers on a device outside the prewarm set were run"


def _groups(params):
    a, b, c = params
    return [
        {"params": [a], "lr": 1e-2, "weight_decay": 0.1},
        {"params": [b], "lr": 4e-2, "weight_decay": 0.0},
        {"params": [c], "lr": 1e-2, "weight_decay": 0.0},
    ]


def _leaves(seed: int):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(8, 8, generator=g)) for _ in range(3)]


def _collapsed(params):
    return [{"params": list(params), "lr": 1e-2, "weight_decay": 0.1}]


def _run(optimizer: str, steps: int = 3, groups=_groups):
    params = _leaves(0)
    bundle = build_bundle(
        groups(params),
        DEV,
        optimizer,
        {"lr": 1e-2, "betas": (0.9, 0.95), "eps": 1e-8},
        {"mode": "cosine", "warmup": 1, "end": 6},
        6,
        build_policy("fp32", ()),
        0.0,
    )
    lrs = []
    for i in range(steps):
        loss = sum(((p - i) ** 2).sum() for p in params)
        bundle.backward(loss)
        assert bundle.step()
        lrs.append([float(g["lr"]) for g in _live_groups(bundle)])
    return params, lrs


def _live_groups(bundle):
    return bundle.optimizer.param_groups


def test_offload_adamw_keeps_group_settings():
    try:
        got, got_lrs = _run("master_adamw")
    except RuntimeError as err:  # fused AdamW unavailable for CPU tensors
        pytest.skip(str(err))
    ref, ref_lrs = _run("adamw")
    assert sum(got_lrs, []) == pytest.approx(sum(ref_lrs, []))
    assert all(lr[1] == pytest.approx(4 * lr[0]) for lr in got_lrs if lr[0] > 0)
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def collapsed():
        flat, _ = _run("master_adamw", groups=_collapsed)
        for a, b in zip(flat, ref):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    must_fail(collapsed, "groups collapsed to one (max wd, base lr)")
