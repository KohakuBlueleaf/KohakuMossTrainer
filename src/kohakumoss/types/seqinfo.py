"""SeqInfo: the packed-batch layout descriptor, built on the host once per batch.

Packed order: image tokens of all samples (sample-major, row-major), and separately text
tokens of all samples. The joint sequence of sample i is `[img_i ; txt_i]`; joint tensors
are laid out sample-major in that order.
"""

from dataclasses import dataclass, field

import torch
from torch import Tensor

from kohakumoss_common.devices import streamed_type
from kohakumoss_kernels.backends.eager.attention import remember_bounds


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

    def pin_memory(self) -> "SeqInfo":
        """Every host tensor (and attention plan boundary) in pinned memory, in place;
        returns self (the HostBatch pin hook)."""
        for name, value in self.__dict__.items():
            match value:
                case Tensor() if value.device.type == "cpu" and not value.is_pinned():
                    setattr(self, name, value.pin_memory())
                case AttnPlan() if not value.cu_seqlens.is_pinned():
                    value.cu_seqlens = value.cu_seqlens.pin_memory()
        return self

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "SeqInfo":
        """Copy every tensor to `device` (host tensors go as one pinned buffer per dtype,
        one copy each; the fields become views); host-side ints and tuples are kept."""
        device = torch.device(device)
        names, tensors = [], []
        for name, value in self.__dict__.items():
            match value:
                case Tensor():
                    names.append((name, None))
                    tensors.append(value)
                case AttnPlan():
                    names.append((name, value))
                    tensors.append(value.cu_seqlens)
        moved_tensors = move_tensors(tensors, device, non_blocking)
        moved = dict(self.__dict__)
        for (name, plan), t in zip(names, moved_tensors):
            if plan is None:
                moved[name] = t
                continue
            remember_bounds(t, _bounds(plan.lengths))
            moved[name] = AttnPlan(t, plan.max_seqlen, plan.lengths, dict(plan.extra))
        return SeqInfo(**moved)


def move_tensors(
    tensors: list[Tensor], device: torch.device, non_blocking: bool = True
) -> list[Tensor]:
    """Move host tensors in one copy per dtype (pinned and asynchronous when the target
    is an accelerator with pinned host memory: CUDA, XPU)."""
    if device.type == "cpu" or any(t.device.type != "cpu" for t in tensors):
        return [t.to(device, non_blocking=non_blocking) for t in tensors]
    pin = streamed_type(device.type)
    out: list[Tensor | None] = [None] * len(tensors)
    groups: dict[torch.dtype, list[int]] = {}
    for i, t in enumerate(tensors):
        groups.setdefault(t.dtype, []).append(i)
    for idx in groups.values():
        flat = torch.cat([tensors[i].reshape(-1) for i in idx])
        flat = flat.pin_memory() if pin else flat
        dev = flat.to(device, non_blocking=non_blocking and pin)
        for i, part in zip(idx, dev.split([tensors[i].numel() for i in idx])):
            out[i] = part.view(tensors[i].shape)
    return out


def _bounds(lengths) -> list[int]:
    out = [0]
    for n in lengths:
        out.append(out[-1] + int(n))
    return out


def _cu(lengths: list[int]) -> Tensor:
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    return cu


def _plan(lengths: list[int]) -> AttnPlan:
    return AttnPlan(_cu(lengths), max(lengths, default=0), tuple(lengths))


def build_seqinfo(
    hw: Tensor,
    text_len: Tensor,
    rope_axes: int = 4,
    img_keep: list[Tensor | None] | None = None,
) -> SeqInfo:
    """Host-side SeqInfo from latent grid sizes `hw` [B, 2] and text lengths [B].

    `img_keep[i]` (sorted int64 indices into sample i's row-major grid, or None for all)
    keeps only those image tokens; their positions are their original grid positions.
    """
    hw = hw.to(torch.int64).cpu()
    text_len = text_len.to(torch.int64).cpu()
    batch = hw.shape[0]
    keeps = img_keep if img_keep is not None else [None] * batch
    img_lens = [
        int(hw[i, 0] * hw[i, 1]) if keeps[i] is None else int(keeps[i].numel())
        for i in range(batch)
    ]
    txt_lens = text_len.tolist()
    joint_lens = [a + b for a, b in zip(img_lens, txt_lens)]

    pos_joint, pos_txt = [], []
    img_in_joint, txt_in_joint = [], []
    seg_img, seg_txt, seg_joint = [], [], []
    offset = 0
    for i in range(batch):
        h, w = int(hw[i, 0]), int(hw[i, 1])
        t = txt_lens[i]
        n = img_lens[i]
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        img = torch.zeros(h * w, rope_axes, dtype=torch.float32)
        img[:, 1] = t
        img[:, 2] = ys.flatten().float() - (h - 1) / 2
        img[:, 3] = xs.flatten().float() - (w - 1) / 2
        if keeps[i] is not None:
            img = img.index_select(0, keeps[i].to(torch.int64))
        txt = torch.zeros(t, rope_axes, dtype=torch.float32)
        txt[:, 1] = torch.arange(t, dtype=torch.float32)
        pos_joint += [img, txt]
        pos_txt.append(txt)
        img_in_joint.append(torch.arange(offset, offset + n))
        txt_in_joint.append(torch.arange(offset + n, offset + n + t))
        offset += n + t
        seg_img.append(torch.full((n,), i, dtype=torch.int32))
        seg_txt.append(torch.full((t,), i, dtype=torch.int32))
        seg_joint.append(torch.full((n + t,), i, dtype=torch.int32))

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
