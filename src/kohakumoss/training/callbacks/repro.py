"""The non-finite step recorder: the last prepared micro-batch of the step that stops a
run (NONFINITE_PATIENCE) and the adapter values, written to `<dump>/repro_step.pt` in
the format tests/training/checkpoint/test_repro_step_nan.py reads (KOHAKUMOSS_REPRO_STEP). See
docs/concepts/training.md#non-finite-steps."""

from pathlib import Path

import torch
from torch import Tensor

FILE = "repro_step.pt"


def kept_indices(seq) -> list[Tensor | None]:
    """Each sample's kept image-token indices into its row-major grid, from the
    layout's image positions (None: the whole grid)."""
    hw = seq.hw.cpu().to(torch.int64)
    cu = seq.cu_img.cpu().tolist()
    pos = seq.pos_joint.cpu()[seq.img_in_joint.cpu()]
    out = []
    for i in range(hw.shape[0]):
        h, w = int(hw[i, 0]), int(hw[i, 1])
        rows = pos[cu[i] : cu[i + 1]]
        if rows.shape[0] == h * w:
            out.append(None)
            continue
        ys = torch.round(rows[:, 2] + (h - 1) / 2).to(torch.int64)
        xs = torch.round(rows[:, 3] + (w - 1) / 2).to(torch.int64)
        out.append(ys * w + xs)
    return out


def repro_record(p, denom: Tensor, adapters: dict[str, Tensor], step: int) -> dict:
    """The record of prepared micro-batch `p` (host tensors)."""
    seq = p.seq
    return {
        "step": int(step),
        "z": p.z.detach().cpu(),
        "text": p.text.detach().cpu(),
        "sigma": p.sigma.detach().cpu(),
        "hw": seq.hw.cpu(),
        "text_len": seq.text_len.cpu(),
        "img_keep": kept_indices(seq),
        "target": p.target.detach().cpu(),
        "weight": p.weight.detach().cpu(),
        "denom": int(denom.reshape(-1)[0]),
        "adapters": {k: v.detach().cpu() for k, v in adapters.items()},
    }


class StepRecorder:
    """`keep(prepared, denom)` references the step's last prepared micro-batch (TrainStep
    calls it) and `drop()` releases it after a finite step; `write(dump_dir, step,
    names)` saves it with the trainable values: every trainable tensor (adapters), or
    with `only_named` (full finetune) just the parameters `names` (the step's
    non-finite report). The values are the ones after the step: a skipped (non-finite)
    step left them as its forward saw them."""

    def __init__(self, trainable, only_named: bool = False):
        self.trainable = trainable
        self.only_named = only_named
        self.last = None

    def keep(self, p, denom: Tensor) -> None:
        self.last = (p, denom)

    def drop(self) -> None:
        self.last = None

    def write(self, dump_dir: str, step: int, names=()) -> Path | None:
        if self.last is None:
            return None
        p, denom = self.last
        named = dict(self.trainable.named_trainable())
        if self.only_named:
            wanted = set(names)
            named = {k: v for k, v in named.items() if k in wanted}
        path = Path(dump_dir) / FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(repro_record(p, denom, named, step), path)
        return path
