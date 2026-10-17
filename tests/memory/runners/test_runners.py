"""Checkpointing runners give the plain runner's loss and gradients."""

import pytest
from torch import nn

from kohakumoss.training.memory.runners import (
    CheckpointRunner,
    PlainRunner,
    StageRunner,
    build_runner,
)
from tests._util.negative import must_fail
from tests.parallel._common import batch, loss_and_grads, make_model, max_diff

pytestmark = pytest.mark.cpu


def _run(make_runner):
    model = make_model()
    model.runner = make_runner(model.layers)
    return loss_and_grads(model, batch())


class _DetachRunner:
    def run(self, index, block: nn.Module, x, ctx):
        return block(x.detach() if index == 1 else x, ctx)


@pytest.mark.parametrize(
    "value", ["block", "sub-block", "selective", "partial:0.5", "none"]
)
def test_checkpoint_matches_plain(value):
    ref_loss, ref = _run(lambda layers: PlainRunner())
    loss, grads = _run(lambda layers: build_runner(value, layers))
    assert abs(loss - ref_loss) < 1e-12
    assert max_diff(grads, ref) < 1e-10


def test_check_catches_a_runner_that_cuts_the_graph():
    _, ref = _run(lambda layers: PlainRunner())

    def check():
        _, grads = _run(lambda layers: _DetachRunner())
        assert grads.keys() == ref.keys() and max_diff(grads, ref) < 1e-10

    must_fail(check, "detached block input")


def test_partial_checkpoints_only_the_first_blocks():
    model = make_model()
    runner = build_runner("partial:0.5", model.layers)
    assert runner.blocks == frozenset({0, 1})
    assert runner._fns[2] is model.layers[2] and runner._fns[3] is model.layers[3]


def test_stage_runner_offsets_index():
    seen = []

    class Rec:
        def bind(self, layers):
            return self

        def run(self, index, block, x, ctx):
            seen.append(index)
            return x

    StageRunner(Rec(), 10).bind(()).run(2, None, 0, None)
    assert seen == [12]


def test_sub_block_binds_block_parts():
    model = make_model()
    runner = CheckpointRunner("sub-block").bind(model.layers)
    assert all(fn.func.__name__ == "_run_parts" for fn in runner._fns)
