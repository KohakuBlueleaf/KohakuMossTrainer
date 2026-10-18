"""The transformers modules' own forward (the reference path): one prompt per call
(`single`) or right-padded rows with an attention mask (`padded`)."""

import importlib.util

import torch
from torch import Tensor, nn

from kohakumoss_common.calls import nothing

HUB_KERNELS = tuple(m for m in ("fla", "causal_conv1d") if importlib.util.find_spec(m))


def _has_delta_layers(text_model: nn.Module) -> bool:
    return any(
        hasattr(layer, "linear_attn") for layer in getattr(text_model, "layers", ())
    )


class HfForward:
    """Final hidden states of every pack through `text_model.forward`; token ids go in
    as embeddings looked up by the placement when its table lives on the host. Raises
    for a Gated DeltaNet model on the host while transformers' GPU-only hub kernels
    (flash-linear-attention / causal-conv1d) are importable: transformers binds them
    into its forward with no device check."""

    def __init__(self, text_model: nn.Module, placement, pad_id: int):
        if (
            placement.device.type == "cpu"
            and HUB_KERNELS
            and _has_delta_layers(text_model)
        ):
            raise ValueError(
                f"text-encoder forward='hf' on the host with {HUB_KERNELS} installed "
                "runs their GPU kernels on host tensors; use forward='kernels'"
            )
        self.text = text_model
        self.placement = placement
        self.pad_id = pad_id
        self.host_embed = placement.name in ("offload", "ring")

    def _run(self, ids: Tensor, mask: Tensor | None) -> Tensor:
        dev = self.placement.device
        kw = {"attention_mask": None if mask is None else mask.to(dev)}
        if self.host_embed:
            emb = self.placement.embed(ids.flatten()).view(*ids.shape, -1)
            out = self.text(inputs_embeds=emb, use_cache=False, return_dict=True, **kw)
        else:
            out = self.text(
                input_ids=ids.to(dev), use_cache=False, return_dict=True, **kw
            )
        return out.last_hidden_state

    unit = "pack"

    def units(self, packs) -> int:
        """Progress units of one call: its packs."""
        return len(packs)

    def __call__(self, packs, tick=nothing) -> list[Tensor]:
        """`tick()` after each pack."""
        out = []
        for p in packs:
            rows = len(p.lengths)
            if p.take is None and rows == 1:
                out.append(self._run(p.ids.view(1, -1), None)[0])
                tick()
                continue
            width = p.max_len
            ids = p.ids.view(rows, width)
            mask = torch.zeros(rows, width, dtype=torch.long)
            for i, n in enumerate(p.lengths):
                mask[i, :n] = 1
            hidden = self._run(ids, mask)
            out.append(torch.cat([hidden[i, :n] for i, n in enumerate(p.lengths)]))
            tick()
        return out
