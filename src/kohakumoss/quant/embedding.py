"""Frozen int8 token-embedding table (a ComfyUI `int8_tensorwise` embedding): rows
q [V, D] int8 times the per-row `scale` [V], stored rotated over D (`rotation`, ConvRot:
row = (q * s) R). A lookup gathers the int8 rows, dequantizes them in fp32 and rotates
them back, then rounds once to `dtype`. See docs/internals/quant-kernels.md#embedding.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kohakumoss.quant.weights import dequantize, unrotated
from kohakumoss_kernels import registry
from kohakumoss_kernels.hadamard import group_hadamard_eager


class QuantEmbedding(nn.Module):
    """nn.Embedding replacement over a stored int8 table (buffers `q`, `scale`)."""

    def __init__(self, q: Tensor, scale: Tensor, rotation: int, dtype: torch.dtype):
        super().__init__()
        self.num_embeddings, self.embedding_dim = q.shape
        self.rotation = int(rotation)
        self.dtype = dtype
        self.register_buffer("q", q)
        self.register_buffer("scale", scale.float().reshape(-1))
        self.rotate = group_hadamard_eager

    def bind(self, dtype: torch.dtype, device: torch.device | str | None = None):
        """The output dtype and the fp32 rotation op of the compute device."""
        self.dtype = dtype
        device = self.q.device if device is None else torch.device(device)
        if self.rotation:
            self.rotate = registry.resolve("group_hadamard", device, torch.float32)

    def finish(self, rows: Tensor, scales: Tensor) -> Tensor:
        """Gathered int8 rows [..., D] and their scales [...] -> embeddings."""
        x = rows.float() * scales.float().unsqueeze(-1)
        if self.rotation:
            x = self.rotate(x, self.rotation)
        return x.to(self.dtype)

    def forward(self, ids: Tensor) -> Tensor:
        return self.finish(F.embedding(ids, self.q), self.scale[ids])

    def lookup(self, ids: Tensor, device: torch.device) -> Tensor:
        """Embeddings on `device` of ids gathered where the table lives (the host under
        offload / ring): only the int8 rows and their scales cross the bus."""
        here = ids.to(self.q.device)
        rows = F.embedding(here, self.q).to(device, non_blocking=True)
        return self.finish(rows, self.scale[here].to(device, non_blocking=True))

    def weight(self, dtype: torch.dtype) -> Tensor:
        """The whole table [V, D] in `dtype` (the rotation undone)."""
        w = dequantize(self.q, self.scale, torch.float32)
        return unrotated(w, self.rotation, dtype)

    def materialize(self) -> Tensor:
        return self.weight(torch.float32)

    def nbytes(self) -> int:
        return self.q.nbytes + self.scale.nbytes


def dense_lookup(
    embedding: nn.Embedding,
    ids: Tensor,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """`QuantEmbedding.lookup` of a dense table: rows gathered where it lives, then
    moved to `device` in `dtype` (default the table's)."""
    weight = embedding.weight
    rows = F.embedding(ids.to(weight.device), weight)
    return rows.to(device, dtype or weight.dtype)
