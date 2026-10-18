"""Every trainable set declares `prewarm_stats` (types.protocols.AdapterSet), and the
adapter prewarm reads it on a full finetune's trainable, model-free (an empty module,
no device)."""

import ast
import inspect
import textwrap

import pytest
import torch
from torch import nn

from kohakumoss.adapters.train.adapter_set import LycoAdapterSet
from kohakumoss.training.full import FullTrainable
from kohakumoss.tuning.prewarm import AdapterPrewarm
from kohakumoss.types import protocols
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
IMPLEMENTATIONS = (LycoAdapterSet, FullTrainable)


def _assigns_stats(cls) -> bool:
    """`cls.__init__` assigns `self.prewarm_stats` (an assignment, not a mention)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__init__)))
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for t in targets:
            if (
                isinstance(t, ast.Attribute)
                and t.attr == "prewarm_stats"
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                return True
    return False


def assert_declared(classes) -> None:
    assert "prewarm_stats" in protocols.AdapterSet.__annotations__
    for cls in classes:
        assert _assigns_stats(cls), cls.__name__


def assert_prewarm_reads(trainable) -> None:
    assert hasattr(trainable, "prewarm_stats"), type(trainable).__name__
    assert trainable.prewarm_stats == {}
    try:
        measured = AdapterPrewarm(trainable, torch.float16, [256])([])
    except AttributeError as err:
        raise AssertionError(f"prewarm read failed: {err}") from err
    assert measured == 0


class _Undeclared(FullTrainable):
    """A trainable without the attribute."""

    def __init__(self, model):
        super().__init__(model)
        del self.prewarm_stats


def test_every_trainable_declares_prewarm_stats():
    assert_declared(IMPLEMENTATIONS)
    must_fail(lambda: assert_declared((_Undeclared,)), "implementation without it")


def test_adapter_prewarm_on_a_full_trainable():
    assert_prewarm_reads(FullTrainable(nn.Module()))
    must_fail(lambda: assert_prewarm_reads(_Undeclared(nn.Module())), "missing stats")
