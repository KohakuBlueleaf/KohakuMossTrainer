"""Pipeline micro-batch layouts: static padding plus token drop (`img_keep`).

Positions of kept image tokens are checked against their grid coordinates computed
here from the kept indices, not against `build_seqinfo`.
"""

import pytest
import torch

from kohakumoss.parallel.pipeline import (
    MicroBatch,
    PipeShape,
    StageModule,
    padded_seqinfo,
)
from kohakumoss.training.memory.synthetic import SyntheticModel
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

HW = [(3, 4), (2, 5)]
TEXT = [3, 2]
KEEP = [torch.tensor([0, 5, 6, 11]), None]
SHAPE = PipeShape(tokens=40, max_samples=3, dim=8, microbatches=1)


def _mb(keep):
    n_img = sum(
        h * w if k is None else k.numel()
        for (h, w), k in zip(HW, keep or [None] * len(HW))
    )
    z = torch.zeros
    return MicroBatch(
        img=z(n_img, 4),
        text=z(sum(TEXT), 4),
        sigma=z(len(HW)),
        hw=torch.tensor(HW),
        text_len=torch.tensor(TEXT),
        target=z(n_img, 4),
        weight=torch.ones(len(HW)),
        img_keep=keep,
    )


def _expected_positions(keep) -> list[tuple[float, float, float]]:
    """(text_len, y, x) of every real image token in packed order."""
    out = []
    for (h, w), t, k in zip(HW, TEXT, keep or [None] * len(HW)):
        idx = range(h * w) if k is None else k.tolist()
        out += [(t, i // w - (h - 1) / 2, i % w - (w - 1) / 2) for i in idx]
    return out


def _check(seq, keep):
    want = _expected_positions(keep)
    real = len(want) + sum(TEXT)
    assert seq.n_joint == SHAPE.tokens
    assert seq.batch == SHAPE.max_samples + 1
    img_rows = seq.img_in_joint[: len(want)]
    got = seq.pos_joint[img_rows][:, 1:4].tolist()
    assert got == [list(p) for p in want]
    assert int((seq.seg_joint < len(HW)).sum()) == real
    assert seq.attn_joint.lengths[: len(HW)] == tuple(
        n + t
        for n, t in zip(
            [
                h * w if k is None else k.numel()
                for (h, w), k in zip(HW, keep or [None] * len(HW))
            ],
            TEXT,
        )
    )


@pytest.mark.parametrize("keep", [None, KEEP], ids=["full", "token_drop"])
def test_padded_seqinfo_layout(keep):
    _check(padded_seqinfo(_mb(keep), SHAPE), keep)


def test_padded_seqinfo_check_catches_ignored_keep():
    def check():
        mb = _mb(KEEP)
        _check(
            padded_seqinfo(MicroBatch(**{**mb.__dict__, "img_keep": None}), SHAPE), KEEP
        )

    must_fail(check, "token drop ignored by the layout")


def test_img_counts_follow_token_drop():
    assert _mb(KEEP).img_counts().tolist() == [4, 10]
    assert _mb(None).img_counts().tolist() == [12, 10]


PIPE_KNOBS = dict(
    BATCHING="tokens",
    MAX_TOKENS=4096,
    MAX_SAMPLES=6,
    TOKEN_BUDGET="exact",
    TEXT_CACHE="offline",
    MICROBATCHES=4,
)


def test_pipe_shape_comes_from_the_data_knobs():
    shape = PipeShape.from_knobs(PIPE_KNOBS, dim=64, rope_axes=4)
    assert (shape.tokens, shape.max_samples, shape.microbatches) == (4096, 6, 4)
    assert shape.dim == 64


@pytest.mark.parametrize(
    "bad",
    [
        {"MAX_SAMPLES": 0},
        {"TOKEN_BUDGET": "estimate"},
        {"BATCHING": "count"},
        {"TEXT_CACHE": "online"},
    ],
    ids=["no_sample_cap", "estimated_budget", "count_batching", "online_text"],
)
def test_pipe_shape_rejects_knobs_without_hard_bounds(bad):
    with pytest.raises(ValueError, match="pipeline micro-batch shape"):
        PipeShape.from_knobs({**PIPE_KNOBS, **bad}, dim=64)


def test_stage_index_is_host_side():
    """The micro-batch index never reads the passed-through device tensor (a meta
    tensor here, which has no value to read)."""
    stage = StageModule(SyntheticModel(16, 32, 2, 4), 0, 2, True, True)
    stage.set_step([None, None], [None, None])
    mb = torch.zeros(1, dtype=torch.int64, device="meta")
    assert [stage._index(mb), stage._index(mb)] == [0, 1]
    with pytest.raises(RuntimeError, match="beyond"):
        stage._index(mb)


def test_padded_seqinfo_rejects_overflow():
    small = PipeShape(tokens=10, max_samples=3, dim=8, microbatches=1)
    with pytest.raises(ValueError, match="exceeds pipe shape"):
        padded_seqinfo(_mb(KEEP), small)
