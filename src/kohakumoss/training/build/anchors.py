"""WD_TARGET: what weight decay pulls the trainable weights towards. 'zero' is the stock
decoupled decay; 'start' decays towards the weights the run started from (WEIGHTS, or
the INIT_FROM checkpoint), held as a bf16 copy on each parameter's device. See
docs/concepts/training.md#optimisers."""

import os

import torch
from torch import Tensor

from kohakumoss.parallel.fsdp import shard_like
from kohakumoss.training.checkpoint.io import load_trainable

KNOB = "WD_TARGET"
TARGETS = ("zero", "start")
ANCHOR_DTYPE = torch.bfloat16
ANCHOR_OPTIMIZERS = ("adamw", "adamw16", "adamw8bit", "adamw8bit_kernel", "muon")


def wd_target_problems(k: dict, full: bool) -> list[str]:
    """WD_TARGET 'start': a full finetune with weight decay and an optimiser whose decay
    is the decoupled p -= lr * wd * p."""
    target = k.get(KNOB, "zero")
    if target not in TARGETS:
        return [f"{KNOB}={target!r} not in {TARGETS}"]
    if target == "zero":
        return []
    p = []
    if not full:
        p.append(f"{KNOB}='start' trains the base weights (ALGO 'full')")
    if float(k["WEIGHT_DECAY"]) <= 0:
        p.append(f"{KNOB}='start' needs WEIGHT_DECAY > 0")
    if k["OPTIMIZER"] not in ANCHOR_OPTIMIZERS:
        p.append(f"{KNOB}='start' runs with OPTIMIZER in {ANCHOR_OPTIMIZERS}")
    return p


@torch.no_grad()
def start_anchors(k: dict, trainable, resumed: bool) -> dict[str, Tensor] | None:
    """Name -> start value of every trainable parameter (None for WD_TARGET 'zero'):
    the live weights before a resume restores them (WEIGHTS after INIT_FROM), or on a
    resumed run with INIT_FROM that checkpoint's trainable state."""
    if k.get(KNOB, "zero") != "start":
        return None
    live = trainable.state_dict()
    if not (resumed and k.get("INIT_FROM")):
        return {n: t.detach().to(ANCHOR_DTYPE, copy=True) for n, t in live.items()}
    init = k["INIT_FROM"]
    if not os.path.isfile(os.path.join(init, "trainable.safetensors")):
        raise SystemExit(
            f"{KNOB}='start' on a resumed run reads its anchors from INIT_FROM={init!r}, "
            "which is not a checkpoint dir (no trainable.safetensors)"
        )
    start = shard_like(load_trainable(init), live)
    return {
        n: start[n].to(device=t.device, dtype=ANCHOR_DTYPE) for n, t in live.items()
    }
