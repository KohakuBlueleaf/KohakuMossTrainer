"""The non-finite step recorder, model-free (explicit tensors): the kept image tokens
come back from the layout as the indices that built it, the written record holds what
tests/training/checkpoint/test_repro_step_nan.py reads, and the guard that stops a run writes it
first and names it."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.training.callbacks import repro
from kohakumoss.training.callbacks.finite import NonFiniteGuard
from kohakumoss.training.callbacks.repro import StepRecorder, kept_indices
from kohakumoss.training.state import TrainState
from kohakumoss.training.step import Prepared
from kohakumoss.types.seqinfo import build_seqinfo
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
HW = torch.tensor([[3, 4], [2, 5], [2, 2]])
TEXT = torch.tensor([3, 2, 4])
KEEP = [torch.tensor([0, 2, 5, 11]), None, torch.tensor([1, 3])]


def _same_keep(got, want) -> None:
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert (g is None) == (w is None), (g, w)
        if w is not None:
            assert torch.equal(g, w), (g, w)


def test_kept_indices_round_trip():
    seq = build_seqinfo(HW, TEXT, img_keep=KEEP)
    _same_keep(kept_indices(seq), KEEP)
    rebuilt = build_seqinfo(HW, TEXT, img_keep=kept_indices(seq))
    assert torch.equal(rebuilt.pos_joint, seq.pos_joint)
    full = kept_indices(build_seqinfo(HW, TEXT))
    assert full == [None, None, None]
    must_fail(lambda: _same_keep(full, KEEP), "a full grid read as the kept subset")


def _prepared() -> Prepared:
    seq = build_seqinfo(HW, TEXT, img_keep=KEEP)
    n_img, n_txt = seq.n_img, seq.n_txt
    return Prepared(
        torch.randn(n_img, 16),
        torch.randn(n_txt, 8),
        torch.rand(3),
        seq,
        torch.randn(n_img, 16),
        torch.ones(3),
        None,
    )


def test_record_and_guard(tmp_path):
    adapter = torch.nn.Parameter(torch.full((2, 2), 3.0))
    trainable = SimpleNamespace(named_trainable=lambda: [("a.lora_up", adapter)])
    recorder = StepRecorder(trainable)
    p = _prepared()
    recorder.keep(p, torch.tensor([42]))
    guard = NonFiniteGuard(1, 1.0, str(tmp_path), recorder.write)
    with pytest.raises(FloatingPointError) as err:
        guard.on_step_end(
            TrainState(step=7, ok=False, nonfinite_params=1, scale=1.0), None
        )
    path = tmp_path / repro.FILE
    assert str(path) in str(err.value)
    rec = torch.load(path)
    assert rec["denom"] == 42 and rec["step"] == 7
    assert torch.equal(rec["z"], p.z) and torch.equal(rec["target"], p.target)
    assert torch.equal(rec["hw"], HW) and torch.equal(rec["text_len"], TEXT)
    _same_keep(rec["img_keep"], KEEP)
    assert torch.equal(rec["adapters"]["a.lora_up"], adapter.detach())
    quiet = NonFiniteGuard(1, 1.0, str(tmp_path / "none"))
    with pytest.raises(FloatingPointError) as err2:
        quiet.on_step_end(
            TrainState(step=7, ok=False, nonfinite_params=1, scale=1.0), None
        )
    must_fail(lambda: _named_in(path.name, str(err2.value)), "no record written")


def _named_in(name: str, text: str) -> None:
    assert name in text, text


def check_forgets(guard: NonFiniteGuard, recorder: StepRecorder) -> None:
    """A finite step releases the kept micro-batch."""
    recorder.keep(_prepared(), torch.tensor([1]))
    guard.on_step_end(TrainState(step=3, ok=True, nonfinite_params=0, scale=1.0), None)
    assert recorder.last is None, "the micro-batch outlived a finite step"


def test_finite_step_drops_the_batch(tmp_path):
    trainable = SimpleNamespace(named_trainable=lambda: [])
    recorder = StepRecorder(trainable)
    check_forgets(
        NonFiniteGuard(2, 1.0, str(tmp_path), recorder.write, recorder.drop), recorder
    )
    held = StepRecorder(trainable)
    must_fail(
        lambda: check_forgets(NonFiniteGuard(2, 1.0, str(tmp_path), held.write), held),
        "a guard that keeps the batch",
    )


def check_named_only(recorder: StepRecorder, tmp_path) -> None:
    """Full finetune writes only the parameters the step reports non-finite."""
    recorder.keep(_prepared(), torch.tensor([1]))
    path = recorder.write(str(tmp_path), 5, ["block.0.w"])
    assert set(torch.load(path)["adapters"]) == {"block.0.w"}


def test_full_finetune_writes_the_named_parameters(tmp_path):
    params = [("block.0.w", torch.ones(2)), ("block.1.w", torch.zeros(2))]
    trainable = SimpleNamespace(named_trainable=lambda: params)
    check_named_only(StepRecorder(trainable, only_named=True), tmp_path / "a")
    must_fail(
        lambda: check_named_only(StepRecorder(trainable), tmp_path / "b"),
        "the whole trainable set written",
    )
