"""COMPILE "block": a block class's forward is compiled once and shared by every instance
(never one compile per block); "off" compiles nothing; a forward called inside a
compiled region runs inline. Model-free: plain modules, torch.compile recorded."""

import pytest
import torch
from torch import nn

from kohakumoss.model import compile as mc
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]


def _block_class(decorate):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        @decorate
        def forward(self, x):
            return self.lin(x)

    return Block


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(fn)
        return fn

    monkeypatch.setattr(mc.torch, "compile", fake_compile)
    yield calls
    mc.set_mode("off")


def _run_blocks(cls, n: int = 3) -> None:
    x = torch.randn(2, 4)
    for block in [cls() for _ in range(n)]:
        block(x)


def check_one_compile(calls: list) -> None:
    assert len(calls) == 1, f"{len(calls)} compiles for one block class"


def test_block_mode_compiles_once_per_class(recorded):
    mc.set_mode("block")
    _run_blocks(_block_class(mc.compiled_block))
    check_one_compile(recorded)


def test_negative_one_compile_per_instance(recorded):
    def per_instance(fn):
        def wrapper(self, *args):
            return torch.compile(fn)(self, *args)

        return wrapper

    mc.set_mode("block")
    _run_blocks(_block_class(per_instance))
    must_fail(lambda: check_one_compile(recorded), "a compile per block")


def test_off_compiles_nothing(recorded):
    mc.set_mode("off")
    _run_blocks(_block_class(mc.compiled_block))
    assert recorded == []


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        mc.set_mode("blocks")
