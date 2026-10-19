"""FSDP_RESHARD: 'forward' reshards each FSDP2 block after its forward (ZeRO-3),
'backward' keeps it gathered through backward (ZeRO-2); the knob reaches every block's
`fully_shard`, is validated, and the memory estimate counts the held blocks
(model-free: recording `fully_shard` / `shard`, the analytic estimate)."""

import importlib

import pytest
from torch import nn

from kohakumoss.memory.profiles import resolve
from kohakumoss.model import resolve_config
from kohakumoss.parallel import fsdp
from kohakumoss.training import knobs
from kohakumoss.training.build import parallel
from kohakumoss.training.build.parallel import fsdp_hooks, reshard_after_forward
from kohakumoss.training.knobs import validate
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

E = importlib.import_module("kohakumoss.memory.estimate")
pytestmark = pytest.mark.cpu


def test_choice_maps_to_the_flag():
    assert reshard_after_forward("forward") is True
    assert reshard_after_forward("backward") is False
    with pytest.raises(ValueError):
        reshard_after_forward("bogus")


def _shard_flags(monkeypatch, flag: bool) -> dict:
    """reshard_after_forward per unit from `fsdp.shard` over recording `fully_shard`."""
    seen: dict[str, bool] = {}

    def record(module, reshard_after_forward, **kw):
        seen[module.tag] = reshard_after_forward

    monkeypatch.setattr(fsdp, "fully_shard", record)
    monkeypatch.setattr(fsdp, "MixedPrecisionPolicy", lambda **kw: None)
    model, own, blocks = nn.Module(), [nn.Module()], [nn.Module(), nn.Module()]
    for tag, m in [("root", model), ("own", own[0]), *zip(("b0", "b1"), blocks)]:
        m.tag = tag
    fsdp.shard(
        model, blocks, own, reshard_after_forward=flag, sum_grads=False, prefetch=0
    )
    return seen


def test_shard_reshards_blocks_only(monkeypatch):
    for flag in (True, False):
        assert _shard_flags(monkeypatch, flag) == {
            "own": False,
            "b0": flag,
            "b1": flag,
            "root": False,
        }


def _hooks_flag(monkeypatch, reshard: str) -> bool:
    """reshard_after_forward that `fsdp_hooks` hands to `shard` for FSDP_RESHARD."""
    got = {}
    monkeypatch.setattr(parallel, "block_units", lambda b: [])
    monkeypatch.setattr(parallel, "conditioning_units", lambda b: [])
    monkeypatch.setattr(parallel, "shard", lambda *a, **kw: got.update(kw))
    fsdp_hooks(object(), 4, "micro", reshard)
    return got["reshard_after_forward"]


def _threads(monkeypatch) -> None:
    assert _hooks_flag(monkeypatch, "forward") is True
    assert _hooks_flag(monkeypatch, "backward") is False


def test_hooks_thread_the_knob(monkeypatch):
    _threads(monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(parallel, "reshard_after_forward", lambda r: True)
        must_fail(lambda: _threads(m), "FSDP_RESHARD not threaded to shard")


def _gathered_bytes(gathered: bool) -> int:
    arch = resolve_config("base")
    plan = resolve("ckpt-resident", {}, "cpu")
    work = E.Workload(
        arch,
        img_tokens=4096,
        txt_tokens=300,
        trainable_params=74_000_000,
        world=4,
        sharded=True,
        gathered=gathered,
    )
    return E.estimate(work, plan).parts.get("gathered", 0)


def _equal(a, b) -> None:
    assert a == b, (a, b)


def test_estimate_counts_held_blocks():
    arch = resolve_config("base")
    blocks = arch.depth * E.block_params(arch)
    assert _gathered_bytes(False) == 0
    assert _gathered_bytes(True) == blocks * 2
    must_fail(lambda: _equal(_gathered_bytes(True), 0), "held blocks not counted")


def _rejects_bogus() -> None:
    bad = [p for p in validate(runnable(FSDP_RESHARD="bogus")) if "FSDP_RESHARD" in p]
    assert bad, "an unknown FSDP_RESHARD passed"


def test_knob_is_validated(monkeypatch):
    _rejects_bogus()
    good = [
        p for p in validate(runnable(FSDP_RESHARD="backward")) if "FSDP_RESHARD" in p
    ]
    assert good == []
    choices = {k: v for k, v in knobs.CHOICES.items() if k != "FSDP_RESHARD"}
    monkeypatch.setattr(knobs, "CHOICES", choices)
    must_fail(_rejects_bogus, "FSDP_RESHARD unchecked")
