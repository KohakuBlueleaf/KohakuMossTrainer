"""A resolved op is freed by reference counting alone, model-free: with the cyclic
collector off, `del` kills a `Bound` (its call path a class function, rebound on a
guard change, still answering like its implementation); a call path bound to the op
keeps it alive."""

import types

import pytest
import torch

from kohakumoss_kernels import registry
from tests._util.negative import must_fail
from tests._util.release import check_released

pytestmark = pytest.mark.cpu
OP = "test_bound_release_square"


@pytest.fixture
def square():
    registry.register(OP, "eager", torch.square)
    yield
    registry._STATE.impls.pop(OP, None)


def _op(bound: bool = False):
    def build() -> tuple:
        op = registry.resolve(OP, "cpu", torch.float32)
        op.guard = not op.guard
        x = torch.randn(4, 8)
        assert torch.equal(op(x), torch.square(x))
        if bound:
            op._call = types.MethodType(registry.Bound._plain, op)
        return (op,)

    return build


def test_bound_op_freed_on_del(square):
    check_released(_op())
    must_fail(lambda: check_released(_op(True)), "a call path bound to the op")
