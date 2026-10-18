"""BlockCtx: what every joint block reads besides its input, and the slot indices of its
per-sample modulation.

Build order (training/entry.py): arch config -> backbone (OpSet + BlockRunner) -> load
weights -> fold (fp16) -> base storage (quantize / stream) -> adapters -> bind_ops ->
parallel wrap -> compile. Every step after `load weights` is a one-time structural change.
"""

from dataclasses import dataclass

from torch import Tensor

from kohakumoss.types.seqinfo import SeqInfo

# Slots of the block modulation [B, 6, D]: attention part, then FFN part.
SHIFT_ATTN, SCALE_ATTN, GATE_ATTN, SHIFT_FFN, SCALE_FFN, GATE_FFN = range(6)
# Slots of the head modulation [B, 2, D].
HEAD_SHIFT, HEAD_SCALE = range(2)


@dataclass
class BlockCtx:
    """Layout, per-sample fp32 modulation [B, 6, D] (slots SHIFT_ATTN .. GATE_FFN), RoPE
    bands, and the residual scale `r` per block (fold)."""

    seq: SeqInfo
    mod: Tensor
    bands: Tensor
    residual_scale: tuple[float, ...]
