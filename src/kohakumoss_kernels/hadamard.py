"""Regular-Hadamard rotation over groups of the last dim (ConvRot).

H_g is the normalised Kronecker power of H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],
[-1,1,1,1]] for g = 4^j: symmetric and involutory, entries +-2^-j (exact in fp16 /
bf16). `group_hadamard(x, g)` is x times blockdiag(H_g) over the last dim, its own
inverse and its own adjoint. See docs/internals/quant-kernels.md#rotation.
"""

from functools import cache

import torch
from torch import Tensor

H4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))
GROUPS = (4, 16, 64, 256, 1024)
HALF = (torch.float16, torch.bfloat16)

__all__ = [
    "GROUPS",
    "H4",
    "check_group",
    "group_hadamard_eager",
    "hadamard_matrix",
    "stages",
]


def stages(group: int) -> int:
    """j of g = 4^j; ValueError unless `group` is a power of 4 of at least 4."""
    j, g = 0, int(group)
    while g > 1 and g % 4 == 0:
        g //= 4
        j += 1
    if g != 1 or j == 0:
        raise ValueError(f"Hadamard group {group} is not a power of 4 >= 4")
    return j


def check_group(x: Tensor, group: int) -> int:
    """The stage count of `group`; ValueError when it does not divide x's last dim."""
    j = stages(group)
    if x.shape[-1] % group:
        raise ValueError(f"last dim {x.shape[-1]} is not a multiple of group {group}")
    return j


@cache
def _matrix(group: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    j = stages(group)
    h4 = torch.tensor(H4, dtype=torch.float32)
    h = h4
    for _ in range(j - 1):
        h = torch.kron(h, h4)
    return (h * 2.0**-j).to(device=device, dtype=dtype)


def hadamard_matrix(
    group: int, dtype: torch.dtype = torch.float32, device="cpu"
) -> Tensor:
    """H_group [group, group] in `dtype` on `device` (cached; do not modify)."""
    return _matrix(int(group), dtype, torch.device(device))


def _traced_matrix(group: int, j: int, dtype: torch.dtype, device) -> Tensor:
    """H_group built by device tensor ops (inside a torch.compile trace, where no
    host-built constant enters the graph): the sign of entry (i, k) is the product over
    the base-4 digits of -1 where digit_i + digit_k == 3."""
    idx = torch.arange(group, device=device)
    i, k = idx[:, None], idx[None, :]
    flips = torch.zeros(group, group, dtype=torch.int32, device=device)
    for d in range(j):
        flips += ((i // 4**d) % 4 + (k // 4**d) % 4 == 3).to(torch.int32)
    sign = 1.0 - 2.0 * (flips % 2).to(torch.float32)
    return (sign * 2.0**-j).to(dtype)


def _butterflies(x: Tensor, group: int, j: int) -> Tensor:
    """x . blockdiag(H_group) by j fp32 4-point stages: y_k = sum(x) - 2 x_(3-k), the
    sum added in index order (the same rounding on every device)."""
    shape = x.shape
    v = x.float()
    rows = v.numel() // group
    for a in range(j):
        x0, x1, x2, x3 = v.reshape(rows * 4**a, 4, group // 4 ** (a + 1)).unbind(1)
        total = ((x0 + x1) + x2) + x3
        v = torch.stack([total - 2.0 * p for p in (x3, x2, x1, x0)], 1)
    return (v.reshape(shape) * 2.0**-j).to(x.dtype)


def group_hadamard_eager(x: Tensor, group: int) -> Tensor:
    """x [..., K] -> x . blockdiag(H_group) [..., K] in x's dtype (autograd through the
    torch ops): a 16-bit matmul with H (fp32 accumulation) for fp16 / bf16, fp32
    4-point butterflies for any other dtype."""
    j = check_group(x, group)
    if x.dtype in HALF:
        if torch.compiler.is_compiling():
            h = _traced_matrix(group, j, x.dtype, x.device)
        else:
            h = hadamard_matrix(group, x.dtype, x.device)
        return (x.reshape(-1, group) @ h).reshape(x.shape)
    return _butterflies(x, group, j)
