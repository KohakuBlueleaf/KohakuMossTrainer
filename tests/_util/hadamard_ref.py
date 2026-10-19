"""Independent fp64 reference of the group Hadamard rotation (torch.kron of the seed)."""

import torch

H4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))


def h64(group: int, sylvester: bool = False, normalise: bool = True) -> torch.Tensor:
    """H_group in fp64 from torch.kron of H4 (or of the 2x2 Sylvester seed)."""
    seed = ((1, 1), (1, -1)) if sylvester else H4
    base = torch.tensor(seed, dtype=torch.float64)
    h = base
    while h.shape[0] < group:
        h = torch.kron(h, base)
    return h / group**0.5 if normalise else h


def rot64(x: torch.Tensor, group: int, **kw) -> torch.Tensor:
    """x . blockdiag(H_group) over the last dim, in fp64 (`kw` as for `h64`)."""
    shape = x.shape
    h = h64(group, **kw).to(x.device)
    return (x.double().reshape(-1, group) @ h).reshape(shape)
