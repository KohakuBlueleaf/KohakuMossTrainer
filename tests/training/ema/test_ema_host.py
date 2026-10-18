"""EMA placement under host masters, model-free (explicit parameters): every average
lives on its own parameter's device (host masters get host averages), the update
first waits for the block optimiser's write-back, and the memory estimate counts the
device averages only."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from kohakumoss.memory.profiles import resolve
from kohakumoss.model import resolve_config
from kohakumoss.training.build import loop
from kohakumoss.training.build.loop import build_ema_for, ema_before_read
from kohakumoss.training.ema import averagers
from kohakumoss.training.ema.averagers import decay_count
from kohakumoss_common.calls import nothing
from tests._util.negative import must_fail

E = importlib.import_module("kohakumoss.memory.estimate")
pytestmark = pytest.mark.cpu


def _built(named):
    return SimpleNamespace(
        trainable=SimpleNamespace(named_trainable=lambda: list(named))
    )


def _named():
    return [
        ("host", torch.nn.Parameter(torch.ones(4))),
        ("meta", torch.nn.Parameter(torch.ones(4, device="meta"))),
    ]


def check_own_device(ema) -> None:
    for tag in ema.tags:
        for buf, p in zip(ema.buffers[tag], ema.params):
            assert buf.device == p.device, (buf.device, p.device)


def test_average_on_its_parameter_device(monkeypatch):
    check_own_device(build_ema_for({"EMA": 0.99}, _built(_named()), None))
    real = averagers.build_ema

    def on_training_device(named, spec, device=None):
        return real(named, spec, device=torch.device("meta"))

    monkeypatch.setattr(loop, "build_ema", on_training_device)
    ema = build_ema_for({"EMA": 0.99}, _built(_named()), None)
    must_fail(lambda: check_own_device(ema), "every average on the training device")


class Blocks:
    def __init__(self, log: list):
        self.log = log

    def synchronize(self) -> None:
        self.log.append("write-back landed")


def check_waits(ema, log: list, monkeypatch) -> None:
    real = torch._foreach_lerp_

    def spy(*args):
        log.append("average")
        return real(*args)

    monkeypatch.setattr(averagers.torch, "_foreach_lerp_", spy)
    ema.update(0)
    getattr(ema, "join", nothing)()
    assert log and log[0] == "write-back landed", log


def test_update_waits_for_the_write_back(monkeypatch):
    log = []
    bundle = SimpleNamespace(blocks=Blocks(log))
    named = [("w", torch.nn.Parameter(torch.ones(3)))]
    ema = build_ema_for({"EMA": 0.9}, _built(named), bundle)
    check_waits(ema, log, monkeypatch)
    assert ema_before_read(SimpleNamespace(blocks=None)) is nothing
    log2 = []
    blind = build_ema_for({"EMA": 0.9}, _built(named), SimpleNamespace(blocks=None))
    must_fail(lambda: check_waits(blind, log2, monkeypatch), "no wait")


def test_decay_count():
    assert decay_count(0) == 0 and decay_count(None) == 0
    assert decay_count(0.999) == 1 and decay_count([0.9, 0.99]) == 2
    assert decay_count({"kind": "power", "sigma_rel": [0.05, 0.1, 0.2]}) == 3
    assert decay_count({"kind": "constant", "decay": 0.9}) == 1


def _ema_part(stream: str, decays: int) -> int:
    arch = resolve_config("base")
    if stream == "swap":
        overrides = {"stream": "swap", "stream_resident": 2, "optim_state": "stream"}
        plan = resolve("manual", overrides, "cpu", trainable_base=True)
    else:
        plan = resolve("ckpt-resident", {}, "cpu", trainable_base=True)
    work = E.Workload(
        arch, img_tokens=4096, txt_tokens=300, full_ft=True, ema_decays=decays
    )
    return E.estimate(work, plan).parts.get("ema", 0)


def test_estimate_counts_device_averages():
    arch = resolve_config("base")
    n, blocks = E.total_params(arch), arch.depth * E.block_params(arch)
    assert _ema_part("off", 2) == 8 * n
    assert _ema_part("swap", 2) == 8 * (n - blocks)
    assert _ema_part("off", 0) == 0
    must_fail(lambda: _equal(_ema_part("swap", 2), 8 * n), "host averages counted")


def _equal(a, b) -> None:
    assert a == b, (a, b)
