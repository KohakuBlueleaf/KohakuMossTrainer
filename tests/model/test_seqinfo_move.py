"""SeqInfo.to: every field arrives intact; host tensors travel as one buffer per dtype."""

import pytest
import torch

from kohakumoss.types.seqinfo import build_seqinfo, move_tensors
from tests._util.negative import must_fail


def _seq():
    keep = [None, torch.tensor([0, 2, 5, 7])]
    return build_seqinfo(
        torch.tensor([[3, 4], [2, 4]]), torch.tensor([5, 2]), img_keep=keep
    )


def _fields(seq):
    out = {k: v for k, v in seq.__dict__.items() if isinstance(v, torch.Tensor)}
    out["cu_joint"] = seq.attn_joint.cu_seqlens
    out["cu_text"] = seq.attn_text.cu_seqlens
    return out


def _same(a, b):
    fa, fb = _fields(a), _fields(b)
    return all(
        torch.equal(fa[k].cpu(), fb[k].cpu()) and fa[k].dtype == fb[k].dtype for k in fa
    )


def _check(cond):
    assert cond


@pytest.mark.cpu
def test_cpu_move_keeps_fields():
    seq = _seq()
    moved = seq.to("cpu")
    assert _same(seq, moved) and moved.attn_joint.lengths == seq.attn_joint.lengths


@pytest.mark.cuda
def test_cuda_move_is_one_buffer_per_dtype():
    seq = _seq()
    moved = seq.to("cuda")
    torch.cuda.synchronize()
    assert _same(seq, moved)
    f = _fields(moved)
    assert (
        f["seg_img"].untyped_storage().data_ptr()
        == f["seg_txt"].untyped_storage().data_ptr()
    )
    assert f["pos_joint"].is_cuda and f["cu_joint"].dtype == torch.int32

    def shuffled(tensors, device, non_blocking=True):
        out = move_tensors(tensors, device, non_blocking)
        return out[1:] + out[:1]

    names = [k for k, v in seq.__dict__.items() if isinstance(v, torch.Tensor)]
    tensors = [seq.__dict__[k] for k in names]
    broken = dict(zip(names, shuffled(tensors, torch.device("cuda"))))
    must_fail(
        lambda: _check(
            all(torch.equal(broken[k].cpu(), seq.__dict__[k]) for k in names)
        ),
        "misplaced views",
    )
