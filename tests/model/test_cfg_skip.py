"""The packed-CFG negative skip and the sampler's layouts, model-free: a segment-skipped
block's output is kept only on unflagged samples (a non-finite value on a flagged one
does not reach it), skip indices are range-checked, and a layout built under inference
mode keeps its cached host bounds."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.diffusion.sampling.sample import layout
from kohakumoss.inference.knobs import DEPTH, skip_problems
from kohakumoss.model.backbone import _run_selected, runs_on
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels.backends.eager.attention import _BOUNDS
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


class NanOnSecond:
    """A block output that is inf / NaN on the second sample's tokens."""

    def __init__(self, n_first: int):
        self.n_first = n_first

    def run(self, index, block, x, ctx):
        out = x + 1.0
        out[self.n_first :] = float("nan")
        return out


def _setup():
    seq = build_seqinfo(torch.tensor([[2, 2], [2, 2]]), torch.tensor([3, 3]))
    x = torch.randn(seq.n_joint, 4)
    mask = torch.tensor([False, True])
    owner = SimpleNamespace(runner=NanOnSecond(7), layers=[None])
    return seq, x, mask, owner


def zero_gates(owner, x, run, r: float) -> torch.Tensor:
    """The gated-residual skip: x * r + block delta * 0 on the flagged samples."""
    out = owner.runner.run(0, None, x, None)
    keep = run.to(x.dtype)
    return x * r * (1 - keep) + out * keep + (out - x) * 0 * (1 - keep)


def _flagged_keep_the_skip(out, x, seq, r: float) -> None:
    second = seq.seg_joint == 1
    assert torch.isfinite(out).all()
    assert torch.equal(out[second], x[second] * r)
    assert torch.equal(out[~second], x[~second] + 1.0)


def _rescaled(model, x, ctx, run):
    return x * 0.5


def test_segment_skip_keeps_the_skip_on_flagged_samples():
    seq, x, mask, owner = _setup()
    run = runs_on(mask, seq)
    assert run.shape == (seq.n_joint, 1) and int(run.sum()) == 7
    out = _run_selected(0, _rescaled, owner, x, None, run)
    _flagged_keep_the_skip(out, x, seq, 0.5)
    must_fail(
        lambda: _flagged_keep_the_skip(zero_gates(owner, x, run, 0.5), x, seq, 0.5),
        "zeroed gates multiply a NaN block output by zero",
    )


def test_skip_indices_are_range_checked():
    def check(problems) -> None:
        assert problems([-1]) and problems([DEPTH]) and problems([0, DEPTH, 3])
        assert problems([0, DEPTH - 1]) == [] and problems(None) == []

    check(skip_problems)
    must_fail(lambda: check(lambda layers, depth=DEPTH: []), "no range check")


def test_layout_built_in_inference_mode_keeps_its_bounds():
    hw, lens = torch.tensor([[2, 3]]), torch.tensor([4])

    def check(build) -> None:
        with torch.inference_mode():
            seq = build(hw, lens, "cpu")
        cu = seq.attn_joint.cu_seqlens
        assert not cu.is_inference()
        assert id(cu) in _BOUNDS

    check(layout)
    must_fail(
        lambda: check(lambda h, t, d: build_seqinfo(h, t).to(d)),
        "an inference-mode layout",
    )
