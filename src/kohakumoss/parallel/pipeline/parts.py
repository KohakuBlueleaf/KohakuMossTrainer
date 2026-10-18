"""A model split into prologue / layers / epilogue for the pipeline engines.

The engines need: `prologue(img, text, sigma, seq) -> (x, PartsCtx)`, `layers`,
`epilogue(x, ctx) -> pred`, `prologue_modules()`, `epilogue_modules()`, `bands`,
`residual_scale`, `residual_dtype`, `conditioning_trainable()`. `BackboneParts` provides
them for MossBackbone.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kohakumoss.types.context import BlockCtx
from kohakumoss.types.seqinfo import SeqInfo


@dataclass
class PartsCtx(BlockCtx):
    """BlockCtx plus the head modulation [B, 2, D] the epilogue needs."""

    head: Tensor | None = None


class BackboneParts(nn.Module):
    """MossBackbone viewed as prologue / layers / epilogue."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.layers = backbone.layers

    @property
    def cfg(self):
        """The backbone's MossArchConfig."""
        return self.backbone.cfg

    @property
    def bands(self) -> Tensor:
        return self.backbone.rope.bands

    @property
    def residual_scale(self) -> tuple[float, ...]:
        return self.backbone.residual_scale

    @property
    def residual_dtype(self) -> torch.dtype:
        """Dtype of the residual stream between blocks (fp32 under the fp32-residual
        fix, else the compute dtype)."""
        return self.backbone._stream_dtype

    def conditioning_modules(self) -> list[nn.Module]:
        bb = self.backbone
        return [bb.time_embedding, bb.shared_block_modulation, bb.video_head_modulation]

    def conditioning_trainable(self) -> bool:
        """Whether the per-sample modulation carries a gradient (a conditioning
        parameter trains)."""
        return any(
            p.requires_grad for m in self.conditioning_modules() for p in m.parameters()
        )

    def prologue(self, img: Tensor, text: Tensor, sigma: Tensor, seq: SeqInfo):
        bb = self.backbone
        mod, head = bb.conditioning(sigma)
        x = bb.embed(img, text, seq)
        return x, PartsCtx(seq, mod, bb.rope.bands, bb.residual_scale, head)

    def epilogue(self, x: Tensor, ctx: PartsCtx) -> Tensor:
        return self.backbone.head(x, ctx.head, ctx.seq)

    def prologue_modules(self) -> list[nn.Module]:
        bb = self.backbone
        return [
            bb.video_embedder,
            bb.text_embedder,
            bb.rope,
            bb.time_embedding,
            bb.shared_block_modulation,
            bb.video_head_modulation,
        ]

    def epilogue_modules(self) -> list[nn.Module]:
        bb = self.backbone
        return [bb.final_norm_video, bb.final_linear_video]
