"""Per-token T-LoRA masks follow LyCORIS's per-timestep rule for each sample."""

import pytest
import torch
from lycoris.modules.tlora import compute_timestep_mask

from kohakumoss.adapters import token_mask
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def test_token_mask_matches_lycoris_rule_per_sample():
    sigma = torch.tensor([0.0, 0.31, 0.77, 1.0])
    seg = torch.tensor([0, 0, 1, 2, 2, 2, 3], dtype=torch.int32)
    mask = token_mask(sigma, seg, max_rank=16, min_rank=2, alpha=1.5)
    for i, s in enumerate(sigma.tolist()):
        want = compute_timestep_mask(int(round(s * 1000)), 1000, 16, 2, 1.5)
        rows = mask[seg == i]
        assert torch.equal(rows, want.expand_as(rows)), i

    def shared_mask():
        flat = token_mask(sigma[:1].expand(4), seg, 16, 2, 1.5)
        assert torch.equal(flat, mask)

    must_fail(shared_mask, "one mask for the whole batch")
