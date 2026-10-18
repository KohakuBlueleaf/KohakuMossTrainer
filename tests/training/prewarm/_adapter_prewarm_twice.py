"""Child process of test_adapter_prewarm_restart: the LoRA autotuned-op prewarm of three
identical linears (no model) on CUDA against KOHAKUMOSS_KERNEL_CACHE; prints one JSON
line {"measured", "timed", "ops", "distinct"}. With argv[1] == "fresh" stored winners
are ignored and identical ops are not deduplicated (the broken variant)."""

import json
import sys

import torch
from torch import nn

from kohakumoss.adapters import AdapterConfig, apply
from kohakumoss.adapters.bind.autotuned import AutotunedOp
from kohakumoss.model.linear import MossLinear
from kohakumoss_kernels.tuning import tuner

OUT, IN, LAYERS = 128, 192, 3


def holder() -> nn.Module:
    """LAYERS identical linears at MOSS module paths (containers, no forward)."""
    root = nn.Module()
    blocks = []
    for _ in range(LAYERS):
        block = nn.Module()
        block.proj = nn.ModuleDict({"q": MossLinear(IN, OUT, bias=True)})
        blocks.append(block)
    root.layers = nn.ModuleList(blocks)
    return root.to("cuda", torch.float16).requires_grad_(False)


def _nothing_stored(self, key):
    return None


def _own(self, dtype, device) -> tuple:
    return (id(self),)


def main() -> None:
    if sys.argv[1:] == ["fresh"]:
        tuner.TuneTable.get = _nothing_stored
        AutotunedOp.signature = _own
    config = AdapterConfig(algo="lora", preset="attn", dim=4, alpha=4.0)
    aset = apply(holder(), config, dtype=torch.float16, device="cuda")
    measured = aset.prewarm([256, 512], torch.float16, [torch.device("cuda", 0)])
    timed = sum(tuner.TIMED.values())
    print(json.dumps({"measured": measured, "timed": timed, **aset.prewarm_stats}))


if __name__ == "__main__":
    main()
