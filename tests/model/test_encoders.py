"""Latent patch layout against an explicit index map (model-free). The prompt template
on the real text encoder runs in tests/integration/test_infer_real.py."""

import pytest
import torch

from kohakumoss.encoders.vae import patchify, unpatchify
from tests._util.negative import must_fail


def _patchify_loop(z):
    b, c, hh, ww = z.shape
    out = torch.empty(b, c * 4, hh // 2, ww // 2)
    for ch in range(c):
        for i in range(2):
            for j in range(2):
                out[:, ch * 4 + i * 2 + j] = z[:, ch, i::2, j::2]
    return out


def _patchify_ji(z):
    """The (j, i) channel order: a layout bug the check must catch."""
    b, c, hh, ww = z.shape
    z = z.reshape(b, c, hh // 2, 2, ww // 2, 2).permute(0, 1, 5, 3, 2, 4)
    return z.reshape(b, c * 4, hh // 2, ww // 2)


def check_layout(patch, unpatch, z) -> None:
    """`patch` equals the index-map loop; `unpatch` inverts both."""
    p = patch(z)
    assert torch.equal(p, _patchify_loop(z)), "patch layout"
    assert torch.equal(unpatch(p), z), "unpatch of patch"
    assert torch.equal(unpatch(_patchify_loop(z)), z), "unpatch of the index map"


@pytest.mark.cpu
def test_patch_layout():
    z = torch.arange(2 * 3 * 6 * 8, dtype=torch.float32).reshape(2, 3, 6, 8)
    check_layout(patchify, unpatchify, z)
    must_fail(lambda: check_layout(_patchify_ji, unpatchify, z), "(j, i) order")
