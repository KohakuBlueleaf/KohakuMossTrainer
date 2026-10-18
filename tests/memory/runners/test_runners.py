"""The stage runner offsets block indices by the stage's first block."""

import pytest

from kohakumoss.memory.runners import StageRunner

pytestmark = pytest.mark.cpu


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
