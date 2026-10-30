"""INIT_FROM loads the checkpoint's trainable state before the optimiser and EMA copy
their masters from the parameters: the first optimiser step starts from the loaded
weights, not the build-time ones; a RESUME checkpoint in the run directory wins over
INIT_FROM. Model-free: plain parameters and a hand-written checkpoint."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from kohakumoss.offload.optim.gpu_adamw import MasterAdamW
from kohakumoss.training.entry import init_weights
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]


class _Trainable(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4, 4))


def _built():
    return SimpleNamespace(
        trainable=_Trainable(), engines=SimpleNamespace(after_restore=lambda: None)
    )


def _ckpt(root, value: float):
    root.mkdir(parents=True)
    save_file({"w": torch.full((4, 4), value)}, str(root / "trainable.safetensors"))
    return root


def _first_step(built, load) -> torch.Tensor:
    """The weight after one zero-gradient master step, `load()` run at its place in
    the build order."""
    load()
    opt = MasterAdamW([built.trainable.w], lr=0.0, weight_decay=0.0)
    built.trainable.w.grad = torch.zeros(4, 4)
    opt.step()
    return built.trainable.w.detach().clone()


def check_loaded(order) -> None:
    w = order()
    assert torch.equal(w, torch.full((4, 4), 3.0)), w


def test_init_from_reaches_the_masters(tmp_path):
    k = {"INIT_FROM": str(_ckpt(tmp_path / "src", 3.0)), "RESUME": "latest"}
    out = tmp_path / "run"
    out.mkdir()

    def built_first():
        built = _built()
        return _first_step(built, lambda: init_weights(k, built, str(out)))

    check_loaded(built_first)

    def loaded_after_optimiser():
        built = _built()
        opt = MasterAdamW([built.trainable.w], lr=0.0, weight_decay=0.0)
        init_weights(k, built, str(out))
        built.trainable.w.grad = torch.zeros(4, 4)
        opt.step()
        return built.trainable.w.detach().clone()

    must_fail(lambda: check_loaded(loaded_after_optimiser), "masters stale")


def test_resume_wins_over_init_from(tmp_path):
    k = {"INIT_FROM": str(_ckpt(tmp_path / "src", 3.0)), "RESUME": "latest"}
    out = tmp_path / "run"
    step = _ckpt(out / "checkpoints" / "step_00000010", 5.0)
    (step / "state.json").write_text(json.dumps({"step": 10}), encoding="utf-8")
    built = _built()
    assert not init_weights(k, built, str(out))
    assert torch.equal(built.trainable.w, torch.zeros(4, 4))
    assert init_weights({**k, "RESUME": ""}, built, str(out))
