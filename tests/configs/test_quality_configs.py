"""The quality-test configs: their run settings, the constant-after-warmup schedule, fp16
by --set, and LoKr's ALGO_KWARGS reaching the adapter configuration unchanged. Every algo
layer's ALGO_KWARGS are LyCORIS arguments."""

import sys
from pathlib import Path

import pytest
import torch

from kohakumoss.adapters.build.config import accepted_kwargs, config_from_knobs
from kohakumoss.cli.config.validate import TRAIN, blocking, compose, problems
from kohakumoss.training.optim.schedule import build_schedule
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
QUALITY = ROOT / "configs" / "moss" / "quality"
ALGO_LAYERS = sorted(
    p
    for p in (ROOT / "configs" / "moss" / "algo").glob("algo_*.py")
    if p.stem != "algo_full"
)
FP16 = {"PRECISION": "fp16", "FP16_FIX": "online", "FP16_FIX_KWARGS": {"c": 0.0625}}


@pytest.fixture(autouse=True)
def linux_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "platform", "linux")


def knobs(name: str, **own) -> dict:
    return compose(ROOT, TRAIN, [QUALITY / f"quality_{name}.py"], own)


@pytest.mark.parametrize("name", ["lora8", "lokr16"])
def test_run_settings(name):
    k = knobs(name)
    assert (k["STEPS"], k["SAVE_EVERY"], k["KEEP_LAST"], k["SEED"]) == (400, 100, 0, 0)
    assert (k["LATENT_CACHE"], k["TEXT_CACHE"]) == ("offline", "online")
    assert k["TEXT_ENCODE"]["memory"] == "ring" and k["MEMORY_PROFILE"] == "auto"
    assert k["OPTIMIZER"] == "adamw" and k["PRECISION"] == "bf16"
    for prec in ({}, FP16):
        assert blocking(problems(TRAIN, {**k, **prec})) == []


@pytest.mark.parametrize(
    "name, algo, dim, alpha, lr",
    [("lora8", "lora", 8, 8.0, 5e-4), ("lokr16", "lokr", 10000, 1.0, 2e-3)],
)
def test_algo_knobs(name, algo, dim, alpha, lr):
    k = knobs(name)
    assert (k["ALGO"], k["DIM"], k["ALPHA"], k["LR"]) == (algo, dim, alpha, lr)


def lr_curve(schedule: dict, lr: float, steps: int) -> list[float]:
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    sched = build_schedule(opt, schedule, steps)
    out = []
    for _ in range(steps):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


def assert_constant_after_warmup(curve: list[float], lr: float, warmup: int) -> None:
    assert curve[0] < lr, curve[:3]
    assert all(abs(v - lr) <= 1e-12 * lr for v in curve[warmup:]), curve[warmup::50]


@pytest.mark.parametrize("name", ["lora8", "lokr16"])
def test_schedule(name):
    k = knobs(name)
    curve = lr_curve(k["SCHEDULE"], k["LR"], k["STEPS"])
    assert_constant_after_warmup(curve, k["LR"], 20)
    cosine = lr_curve({"mode": "cosine", "warmup": 20, "end": -1}, k["LR"], 400)
    must_fail(lambda: assert_constant_after_warmup(cosine, k["LR"], 20), "cosine")


def test_lokr_kwargs_reach_the_adapter_config():
    k = knobs("lokr16")
    assert k["ALGO_KWARGS"] == {"factor": 16, "full_matrix": True}
    config = config_from_knobs(k)
    expected = ("lokr", {"factor": 16, "full_matrix": True})
    assert (config.algo, config.algo_kwargs) == expected


def assert_lycoris_kwargs(k: dict) -> None:
    config = config_from_knobs(k)
    unknown = set(config.algo_kwargs) - accepted_kwargs(config.algo)
    assert not unknown, unknown
    assert config.validate() == [], config.validate()


@pytest.mark.parametrize("layer", ALGO_LAYERS, ids=lambda p: p.stem)
def test_algo_layer_kwargs_are_lycoris_names(layer):
    k = compose(ROOT, TRAIN, [ROOT / "configs/moss/_base/base_lora.py", layer], {})
    assert_lycoris_kwargs(k)
    bad = {**k, "ALGO_KWARGS": {**k["ALGO_KWARGS"], "fullmatrix": True}}
    must_fail(lambda: assert_lycoris_kwargs(bad), "misspelled LyCORIS argument")
