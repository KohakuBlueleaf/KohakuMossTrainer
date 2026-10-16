"""SeqInfo: the packed-batch layout descriptor, built on the host once per batch.

Packed order: image tokens of all samples (sample-major, row-major), and separately text
tokens of all samples. The joint sequence of sample i is `[img_i ; txt_i]`; joint tensors
are laid out sample-major in that order.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class AttnPlan:
    """Segment boundaries for one attention call; `extra` holds backend-built plans."""

    cu_seqlens: Tensor
    max_seqlen: int
    lengths: tuple[int, ...]
    extra: dict = field(default_factory=dict)


@dataclass
class SeqInfo:
    """Layout of a packed batch of B samples."""

    batch: int
    hw: Tensor
    text_len: Tensor
    cu_img: Tensor
    cu_txt: Tensor
    n_img: int
    n_txt: int
    seg_img: Tensor
    seg_txt: Tensor
    seg_joint: Tensor
    pos_txt: Tensor
    pos_joint: Tensor
    img_in_joint: Tensor
    txt_in_joint: Tensor
    attn_joint: AttnPlan
    attn_text: AttnPlan

    @property
    def n_joint(self) -> int:
        return self.n_img + self.n_txt

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "SeqInfo":
        """Copy every tensor to `device`; host-side ints and tuples are kept."""
        moved = {}
        for name, value in self.__dict__.items():
            match value:
                case Tensor():
                    moved[name] = value.to(device, non_blocking=non_blocking)
                case AttnPlan():
                    moved[name] = AttnPlan(
                        value.cu_seqlens.to(device, non_blocking=non_blocking),
                        value.max_seqlen,
                        value.lengths,
                        dict(value.extra),
                    )
                case _:
                    moved[name] = value
        return SeqInfo(**moved)


def _cu(lengths: list[int]) -> Tensor:
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    return cu


def _plan(lengths: list[int]) -> AttnPlan:
    return AttnPlan(_cu(lengths), max(lengths, default=0), tuple(lengths))


def build_seqinfo(hw: Tensor, text_len: Tensor, rope_axes: int = 4) -> SeqInfo:
    """Host-side SeqInfo from latent grid sizes `hw` [B, 2] and text lengths [B]."""
    hw = hw.to(torch.int64).cpu()
    text_len = text_len.to(torch.int64).cpu()
    batch = hw.shape[0]
    img_lens = (hw[:, 0] * hw[:, 1]).tolist()
    txt_lens = text_len.tolist()
    joint_lens = [a + b for a, b in zip(img_lens, txt_lens)]

    pos_joint, pos_txt = [], []
    img_in_joint, txt_in_joint = [], []
    seg_img, seg_txt, seg_joint = [], [], []
    offset = 0
    for i in range(batch):
        h, w = int(hw[i, 0]), int(hw[i, 1])
        t = txt_lens[i]
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        img = torch.zeros(h * w, rope_axes, dtype=torch.float32)
        img[:, 1] = t
        img[:, 2] = ys.flatten().float() - (h - 1) / 2
        img[:, 3] = xs.flatten().float() - (w - 1) / 2
        txt = torch.zeros(t, rope_axes, dtype=torch.float32)
        txt[:, 1] = torch.arange(t, dtype=torch.float32)
        pos_joint += [img, txt]
        pos_txt.append(txt)
        img_in_joint.append(torch.arange(offset, offset + h * w))
        txt_in_joint.append(torch.arange(offset + h * w, offset + h * w + t))
        offset += h * w + t
        seg_img.append(torch.full((h * w,), i, dtype=torch.int32))
        seg_txt.append(torch.full((t,), i, dtype=torch.int32))
        seg_joint.append(torch.full((h * w + t,), i, dtype=torch.int32))

    return SeqInfo(
        batch=batch,
        hw=hw,
        text_len=text_len,
        cu_img=_cu(img_lens),
        cu_txt=_cu(txt_lens),
        n_img=sum(img_lens),
        n_txt=sum(txt_lens),
        seg_img=torch.cat(seg_img),
        seg_txt=torch.cat(seg_txt),
        seg_joint=torch.cat(seg_joint),
        pos_txt=torch.cat(pos_txt),
        pos_joint=torch.cat(pos_joint),
        img_in_joint=torch.cat(img_in_joint),
        txt_in_joint=torch.cat(txt_in_joint),
        attn_joint=_plan(joint_lens),
        attn_text=_plan(txt_lens),
    )
