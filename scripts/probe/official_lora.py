"""The official MOSS-Image transformer (the diffusers code shipped with the weights) with
plain PyTorch LoRA wrappers holding the live adapter values, as a bf16-autocast gradient
baseline for scripts/probe/grad_refs.py.
"""

import importlib.util
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def load_official_module(weights: str):
    """`pipeline_moss_image.py` of the weights directory as a module."""
    path = Path(weights) / "pipeline_moss_image.py"
    spec = importlib.util.spec_from_file_location("official_moss_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoRALinear(nn.Module):
    """base(x) + scale * up(down(x)); base frozen, down / up trainable."""

    def __init__(self, base: nn.Linear, down: torch.Tensor, up: torch.Tensor, scale):
        super().__init__()
        self.base = base
        self.down = nn.Parameter(down.detach().clone().float())
        self.up = nn.Parameter(up.detach().clone().float())
        self.scale = float(scale)

    @property
    def weight(self) -> torch.Tensor:
        """The base weight (the official blocks read its dtype)."""
        return self.base.weight

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.down), self.up) * self.scale


class OfficialLoRA:
    """Official transformer in `dtype` weights on `device` with every adapted joint-block
    linear of `live` (a LycoAdapterSet) wrapped by a LoRALinear holding its values;
    joint blocks are checkpointed (non-reentrant) to fit one 32 GB card."""

    def __init__(self, weights: str, live, device, dtype=torch.bfloat16):
        module = load_official_module(weights)
        model = module.MOSSImageTransformer.from_pretrained(
            str(Path(weights) / "transformer"), torch_dtype=dtype
        )
        model.requires_grad_(False).to(device)
        for block in model.layers:
            block.forward = partial(checkpoint, block.forward, use_reentrant=False)
        self.wrappers = {}
        params = dict(live.named_trainable())
        for t in live.targets:
            layer, name = _path_of(t.lora_name)
            proj = model.layers[layer].proj
            down = params[f"{t.lora_name}.lora_down.weight"]
            up = params[f"{t.lora_name}.lora_up.weight"]
            wrapper = LoRALinear(proj[name], down, up, t.lyco.scale).to(device)
            proj[name] = wrapper
            self.wrappers[t.lora_name] = wrapper
        self.order = [name for name, _ in live.named_trainable()]
        self.model, self.device = model, device

    def grad(self, p, denom, autocast_dtype=torch.bfloat16) -> torch.Tensor:
        """Flat adapter gradient (the live set's parameter order) of prepared batch `p`,
        one sample per forward as the official code runs, under bf16 autocast."""
        for w in self.wrappers.values():
            w.down.grad = w.up.grad = None
        seq, dev = p.seq, self.device
        cu_img, cu_txt = seq.cu_img.tolist(), seq.cu_txt.tolist()
        for i, (h, w) in enumerate(seq.hw.tolist()):
            z = p.z[cu_img[i] : cu_img[i + 1]].to(dev).T.reshape(1, -1, h, w)
            text = p.text[cu_txt[i] : cu_txt[i + 1]].to(dev)[None]
            target = p.target[cu_img[i] : cu_img[i + 1]].to(dev)
            with torch.autocast("cuda", dtype=autocast_dtype):
                out = self.model(z, p.sigma[i : i + 1].to(dev) * 1000.0, text).sample
            pred = out.float()[0].flatten(1).T
            loss = (pred - target.float()).square().mean(-1).sum() / float(denom)
            loss.backward()
        grads = {}
        for lora_name, wrapper in self.wrappers.items():
            grads[f"{lora_name}.lora_down.weight"] = wrapper.down.grad
            grads[f"{lora_name}.lora_up.weight"] = wrapper.up.grad
        return torch.cat(
            [grads[n].detach().flatten().double().cpu() for n in self.order]
        )

    def close(self) -> None:
        self.model.to("meta")
        del self.model, self.wrappers
        torch.cuda.empty_cache()


def _path_of(lora_name: str) -> tuple[int, str]:
    """`lora_unet_layers_3_proj_ffn_up_gate` -> (3, "ffn_up_gate")."""
    rest = lora_name.split("layers_", 1)[1]
    layer, name = rest.split("_proj_", 1)
    return int(layer), name
