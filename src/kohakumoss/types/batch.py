"""DeviceBatch: a HostBatch moved to the compute device with its SeqInfo."""

from dataclasses import dataclass

from torch import Tensor

from kohakumoss.types.seqinfo import SeqInfo


@dataclass
class DeviceBatch:
    """Packed batch on device. `latents` [N_img, C] and `text` [N_txt, text_dim] may be
    None when encoders run online (then `pixels` / `prompts` are set)."""

    ids: Tensor
    seq: SeqInfo
    latents: Tensor | None
    text: Tensor | None
    text_drop: Tensor
    weight: Tensor
    loss_mask: Tensor | None = None
    pixels: list | None = None
    prompts: list[str] | None = None
