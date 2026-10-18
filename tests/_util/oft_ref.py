"""fp64 references of the orthogonal-finetuning kinds (Diag-OFT, BOFT): the Cayley
rotation with its inverse in float64, and BOFT's butterfly on the channel axis."""

import pytest
import torch
from lycoris.functional import boft as f_boft
from lycoris.functional import diag_oft as f_oft

LYCORIS_GET_R = f_oft.get_r


def cayley(blocks: torch.Tensor, constraint: float | None = 0) -> torch.Tensor:
    """(I + Q) (I - Q)^-1 of Q = blocks - blocks^T in the blocks' dtype, Q rescaled to
    norm `constraint` when it is larger (LyCORIS's rule)."""
    eye = torch.eye(blocks.shape[-1], device=blocks.device, dtype=blocks.dtype)
    q = blocks - blocks.transpose(-1, -2)
    if constraint is not None and constraint > 0:
        norm = torch.norm(q) + 1e-8
        if norm > constraint:
            q = q * constraint / norm
    return (eye + q) @ torch.linalg.inv(eye - q)


def butterfly(r: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """BOFT's butterfly of rotations r [m, num, b, b] applied on y's channel axis."""
    m, _, b, _ = r.shape
    out = y
    for i in range(m):
        k = 2**i * (b // 2)
        out = out.unflatten(-1, (-1, 2, k)).transpose(-2, -1).flatten(-3)
        out = torch.einsum(
            "b i j, ... b j -> ... b i", r[i], out.unflatten(-1, (-1, b))
        )
        out = out.flatten(-2).unflatten(-1, (-1, k, 2)).transpose(-2, -1).flatten(-3)
    return out


def get_r64(oft_blocks, eye=None, constraint=0):
    """LyCORIS's get_r; float64 blocks (a reference run) rotate through `cayley` in
    float64 (LyCORIS inverts in float32)."""
    if oft_blocks.dtype != torch.float64:
        return LYCORIS_GET_R(oft_blocks, eye, constraint)
    return cayley(oft_blocks, constraint)


@pytest.fixture
def fp64_cayley(monkeypatch):
    """LyCORIS's OFT / BOFT bodies run their float64 calls through `get_r64`."""
    monkeypatch.setattr(f_oft, "get_r", get_r64)
    monkeypatch.setattr(f_boft, "get_r", get_r64)
