"""Prewarm snapshots, model-free: a snapshot keeps a tensor's layout, keeps small
integer values, and materializes on the device the tensor ran on. The replay of a real
streamed text-encoder pass runs in tests/integration/test_train_real.py."""

import pytest
import torch

from kohakumoss_kernels import recording
from kohakumoss_kernels.recording import Snap

pytestmark = pytest.mark.cuda


def test_snapshot_keeps_layout_and_small_values():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    big = torch.randn(1 << 17, device="cuda", dtype=torch.float16)
    ids = torch.arange(10, device="cuda", dtype=torch.int32)
    q = torch.randint(-5, 6, (1 << 17,), device="cuda", dtype=torch.int8)
    s_big, s_ids, s_q = (recording.snapshot(t) for t in (big, ids, q))
    assert isinstance(s_big, Snap) and s_big.values is None
    assert torch.equal(s_ids.values, ids) and s_q.span == (-5, 5)
    ids.add_(1)
    assert torch.equal(s_ids.tensor(), torch.arange(10, device="cuda").int())
    assert s_big.tensor(7).shape == (7,) and s_big.tensor().device == big.device
    r = s_q.tensor()
    assert r.dtype == torch.int8 and int(r.min()) >= -5 and int(r.max()) <= 5
