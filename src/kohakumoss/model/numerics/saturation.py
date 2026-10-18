"""Debug counters: per-block max |x| and count of values above a threshold (no host sync)."""

import torch

from kohakumoss.model.backbone import MossBackbone


def _recomputing() -> bool:
    """True inside a backward pass (a checkpoint's forward recompute)."""
    return torch._C._current_graph_task_id() != -1


class SaturationCounters:
    """Forward hooks on every joint and text block, counting the original forward only
    (not a checkpoint's recompute); `read()` syncs once and returns stats."""

    def __init__(self, model: MossBackbone, threshold: float = 2.0**15):
        device = next(model.parameters()).device
        blocks = [(f"text.{i}", b) for i, b in enumerate(model.text_embedder.blocks)]
        blocks += [(f"layer.{i}", b) for i, b in enumerate(model.layers)]
        self.threshold = threshold
        self.names = [name for name, _ in blocks]
        self.maxabs = torch.zeros(len(blocks), device=device)
        self.over = torch.zeros(len(blocks), dtype=torch.int64, device=device)
        self.calls = [0] * len(blocks)
        self.handles = [
            block.register_forward_hook(self._hook(i))
            for i, (_, block) in enumerate(blocks)
        ]

    def _hook(self, idx: int):
        def hook(module, args, output):
            if _recomputing():
                return
            a = output.detach().abs()
            self.maxabs[idx] = torch.maximum(self.maxabs[idx], a.max().float())
            self.over[idx] += (a > self.threshold).sum()
            self.calls[idx] += 1

        return hook

    def read(self) -> dict[str, dict[str, float]]:
        maxabs, over = self.maxabs.tolist(), self.over.tolist()
        return {
            name: {"maxabs": maxabs[i], "over": over[i]}
            for i, name in enumerate(self.names)
            if self.calls[i]
        }

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
