"""Checkpoint `.pt` files load with `weights_only=True`: the RNG (numpy state included)
and optimiser files a checkpoint holds load; a pickle carrying any other global is
refused."""

import pickle

import pytest
import torch

from kohakumoss.training.checkpoint import io
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


class Payload:
    """A global no checkpoint writes."""

    def __reduce__(self):
        return (print, ("loaded",))


def test_rng_and_optimiser_files_load(tmp_path):
    gen = torch.Generator().manual_seed(3)
    io.save_rng(tmp_path, 0, io.capture_rng({"sigma": gen.get_state()}))
    rng = io.load_rng(tmp_path, 0)
    assert torch.equal(rng["step"]["sigma"], gen.get_state())
    layer = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(layer.parameters())
    layer(torch.randn(2, 4)).sum().backward()
    opt.step()
    io.save_optim_rank(tmp_path, 0, {"optimizers": [opt.state_dict()], "step": 1})
    assert io.load_optim_rank(tmp_path, 0)["step"] == 1


def check_refuses(load, path) -> None:
    try:
        load(path)
    except pickle.UnpicklingError:
        return
    raise AssertionError("a checkpoint with a foreign global loaded")


def _unsafe(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_a_foreign_global_is_refused(tmp_path):
    path = tmp_path / "optim.pt"
    torch.save({"state": Payload()}, path)
    check_refuses(io.load_pt, path)
    must_fail(lambda: check_refuses(_unsafe, path), "weights_only=False")
