"""Memory-knob combinations through the real trainer (`entry.run`, tiny model,
synthetic data, CPU): after 2 steps every trainable tensor equals the plain run's.

Axes: adapter / full FT x stream (off; ring for adapters, swap for full FT) x activation
offload x gradient offload x fused step x optimiser state x GRAD_ACC. Combinations the
knobs cannot run must be rejected by `validate` with a message.
"""

import itertools

import pytest
import torch
from safetensors.torch import load_file

from kohakumoss.model.opset import OpSet
from kohakumoss.training import entry
from kohakumoss.training.build import loop
from kohakumoss.training.build import model as model_build
from kohakumoss.training.knobs import resolve_precision, validate
from kohakumoss.training.memory import optimizers
from kohakumoss.training.memory.apply import apply_to_backbone
from kohakumoss.training.precision import build_policy
from tests._util.negative import must_fail
from tests.training.test_entry_smoke import smoke_knobs

pytestmark = [pytest.mark.cpu, pytest.mark.slow]

STEPS = 2
TOL = 1e-4
STREAMS = {"lora": ("off", "ring"), "full": ("off", "swap")}


def _combos():
    for algo in ("lora", "full"):
        for st, act, go, fused, opt, acc in itertools.product(
            STREAMS[algo],
            (False, True),
            (False, True),
            (False, True),
            ("gpu", "stream", "cpu"),
            (1, 2),
        ):
            yield algo, st, act, go, fused, opt, acc


def _knobs(tmp, combo, lr_scale=1.0):
    algo, st, act, go, fused, opt, acc = combo
    overrides = dict(
        checkpoint="block",
        stream=st,
        stream_resident=2,
        act_offload=act,
        grad_offload=go,
        fused_step=fused,
        optim_state=opt,
    )
    return smoke_knobs(
        tmp,
        ALGO=algo,
        STEPS=STEPS,
        GRAD_ACC=acc,
        SAVE_EVERY=STEPS,
        GRAD_CLIP=0.0,
        LR=1e-2 * lr_scale,
        SCHEDULE={"mode": "constant"},
        EXPORT_FORMATS=(),
        LOGGER="console",
        MEMORY_PROFILE="balanced",
        MEMORY_OVERRIDES=overrides,
    )


COMBOS = list(_combos())
VALID = [c for c in COMBOS if not validate(_knobs("x", c))]
INVALID = [c for c in COMBOS if validate(_knobs("x", c))]
PLAIN = {"off", False}
_REFS: dict = {}


def _ids(c) -> str:
    return "-".join(str(x) for x in c)


def _final(tmp, knobs) -> dict:
    entry.run(knobs)
    path = tmp / "smoke" / "checkpoints" / f"step_{STEPS:08d}" / "trainable.safetensors"
    return load_file(str(path))


def _init(knobs) -> dict:
    torch.manual_seed(knobs["SEED"])
    policy = build_policy(resolve_precision(knobs), knobs["KEEP_FP32"])
    built = model_build.build_model(knobs, torch.device("cpu"), policy)
    return {
        k: v.detach().float().cpu().clone()
        for k, v in built.trainable.state_dict().items()
    }


def _relative(got: dict, ref: dict, init: dict) -> float:
    scale = max((ref[k].float() - init[k]).abs().max().item() for k in ref)
    worst = max((got[k].float() - ref[k].float()).abs().max().item() for k in ref)
    return worst / scale


def _reference(tmp, algo, acc):
    """(init, plain final, tolerance): the tolerance is TOL or 20x the run-to-run
    spread of the plain run itself (non-deterministic GPU kernels)."""
    key = (algo, acc)
    if key not in _REFS:
        plain = (algo, "off", False, False, False, "gpu", acc)
        knobs = _knobs(tmp / "ref", plain)
        init, ref = _init(knobs), _final(tmp / "ref", knobs)
        again = _final(tmp / "ref2", _knobs(tmp / "ref2", plain))
        _REFS[key] = (init, ref, max(TOL, 20 * _relative(again, ref, init)))
    return _REFS[key]


def _error(tmp, combo, lr_scale=1.0) -> tuple[float, float]:
    """(relative update error vs the plain run, tolerance)."""
    init, ref, tol = _reference(tmp, combo[0], combo[-1])
    got = _final(tmp / "run", _knobs(tmp / "run", combo, lr_scale))
    assert got.keys() == ref.keys()
    moved = [k for k in ref if (ref[k].float() - init[k]).abs().max() > 0]
    still = [k for k in moved if torch.equal(got[k].float(), init[k])]
    assert not still, f"not trained: {still[:4]}"
    return _relative(got, ref, init), tol


@pytest.mark.parametrize("combo", VALID, ids=_ids)
def test_trainer_combination_matches_plain(tmp_path, combo):
    err, tol = _error(tmp_path, combo)
    assert err < tol, (err, tol)


@pytest.mark.parametrize("combo", INVALID, ids=_ids)
def test_trainer_combination_rejected(combo):
    assert any("swap" in m for m in validate(_knobs("x", combo)))


def test_matrix_catches_a_wrong_update(tmp_path):
    combo = ("full", "swap", True, True, True, "stream", 2)
    must_fail(lambda: _assert_close(tmp_path, combo, 1.25), "a 25% larger LR")


def _assert_close(tmp, combo, lr_scale=1.0):
    err, tol = _error(tmp, combo, lr_scale)
    assert err < tol, (err, tol)


def _trainable_before_plan(k, device, policy, world=1):
    """The build order with the trainable set taken before the memory plan."""
    backbone = model_build.build_backbone(k, torch.device("cpu"), policy)
    plan, _ = model_build.resolve_memory(k, device, policy.compute, world)
    trainable = model_build.build_trainable(k, backbone, policy, device)
    engines = apply_to_backbone(
        backbone, plan, device, policy.compute, training=True, trainable_base=True
    )
    backbone.bind_ops(OpSet.resolve(device, policy.compute, k["KERNELS"]))
    encoders = model_build.build_encoders(k, device, policy)
    return model_build.Built(backbone, trainable, plan, engines, *encoders)


def test_matrix_catches_streamed_blocks_outside_the_trainable_set(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(entry, "build_model", _trainable_before_plan)
    combo = ("full", "swap", False, False, False, "cpu", 1)
    must_fail(lambda: _assert_close(tmp_path, combo), "trainable set built before swap")


def test_matrix_catches_host_lanes_under_fused_offload(tmp_path, monkeypatch):
    real = optimizers.build_memory_optim

    def host_lanes(*args, **kwargs):
        bundle, arm = real(*args, **kwargs)
        engines = args[2]
        shared = getattr(bundle, "shared", None)
        if engines.grads is not None and hasattr(shared, "use_host_grads"):
            shared.use_host_grads(engines.grads)
        return bundle, arm

    monkeypatch.setattr(loop, "build_memory_optim", host_lanes)
    combo = ("lora", "off", False, True, True, "cpu", 1)
    must_fail(lambda: _assert_close(tmp_path, combo), "fused CPUAdamW on host lanes")
