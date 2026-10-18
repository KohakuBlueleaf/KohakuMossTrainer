"""Single-file FLUX.2 VAE checkpoints (ComfyUI's `vae/` folder): BFL / ldm keys
(`flux2-vae`) mapped onto AutoencoderKLFlux2's diffusers keys, the AutoencoderKLFlux2
config inferred from tensor shapes (either key style), and the fp32 load of one file.
See docs/guides/weights.md#vae."""

import re
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2
from safetensors import safe_open
from torch import Tensor

BFL_MARK = "encoder.down.0.block.0.conv1.weight"
RESNET_PARTS = {"nin_shortcut": "conv_shortcut"}
ATTN_PARTS = {
    "norm": "group_norm",
    "q": "to_q",
    "k": "to_k",
    "v": "to_v",
    "proj_out": "to_out.0",
}
SIDE_PARTS = {"norm_out": "conv_norm_out", "conv_in": "conv_in", "conv_out": "conv_out"}
TOP_LEVEL = {
    "encoder.quant_conv": "quant_conv",
    "decoder.post_quant_conv": "post_quant_conv",
}
STOCK = {
    "act_fn": "silu",
    "norm_num_groups": 32,
    "sample_size": 1024,
    "batch_norm_eps": 1e-4,
    "batch_norm_momentum": 0.1,
}
_DOWN = re.compile(r"^encoder\.down\.(\d+)\.(block\.(\d+)|downsample)\.(.+)$")
_UP = re.compile(r"^decoder\.up\.(\d+)\.(block\.(\d+)|upsample)\.(.+)$")
_MID_BLOCK = re.compile(r"^(encoder|decoder)\.mid\.block_(\d+)\.(.+)$")
_MID_ATTN = re.compile(r"^(encoder|decoder)\.mid\.attn_1\.([a-z_]+)\.(weight|bias)$")
_SIDE = re.compile(r"^(encoder|decoder)\.([a-z_]+)\.(weight|bias)$")
_TOP = re.compile(r"^(encoder\.quant_conv|decoder\.post_quant_conv)\.(weight|bias)$")


def is_bfl(keys) -> bool:
    """Whether a key set is in the BFL / ldm layout (`flux2-vae`)."""
    return BFL_MARK in keys


def up_blocks(keys) -> int:
    """The number of decoder up blocks of a BFL key set."""
    found = {int(m.group(1)) for k in keys if (m := _UP.match(k))}
    return 1 + max(found) if found else 0


def _resnet(rest: str) -> str:
    part, _, tail = rest.partition(".")
    return f"{RESNET_PARTS.get(part, part)}.{tail}"


def bfl_key(key: str, n_up: int) -> str:
    """The diffusers key of BFL key `key` (`n_up` decoder up blocks); KeyError for a key
    outside the layout."""
    if key.startswith("bn."):
        return key
    if m := _TOP.match(key):
        return f"{TOP_LEVEL[m.group(1)]}.{m.group(2)}"
    if m := _DOWN.match(key):
        i, rest = m.group(1), m.group(4)
        if m.group(3) is None:
            return f"encoder.down_blocks.{i}.downsamplers.0.{rest}"
        return f"encoder.down_blocks.{i}.resnets.{m.group(3)}.{_resnet(rest)}"
    if m := _UP.match(key):
        i, rest = n_up - 1 - int(m.group(1)), m.group(4)
        if m.group(3) is None:
            return f"decoder.up_blocks.{i}.upsamplers.0.{rest}"
        return f"decoder.up_blocks.{i}.resnets.{m.group(3)}.{_resnet(rest)}"
    if m := _MID_BLOCK.match(key):
        index = int(m.group(2)) - 1
        return f"{m.group(1)}.mid_block.resnets.{index}.{_resnet(m.group(3))}"
    if (m := _MID_ATTN.match(key)) and m.group(2) in ATTN_PARTS:
        part = ATTN_PARTS[m.group(2)]
        return f"{m.group(1)}.mid_block.attentions.0.{part}.{m.group(3)}"
    if (m := _SIDE.match(key)) and m.group(2) in SIDE_PARTS:
        return f"{m.group(1)}.{SIDE_PARTS[m.group(2)]}.{m.group(3)}"
    raise KeyError(f"{key!r} is not a FLUX.2 VAE key of the BFL layout")


def _is_attn_linear(key: str) -> bool:
    m = _MID_ATTN.match(key)
    return m is not None and m.group(2) != "norm" and m.group(3) == "weight"


def bfl_shape(key: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    """The diffusers shape of BFL tensor `key`: mid attention 1x1 convs become linears."""
    return shape[:2] if _is_attn_linear(key) else shape


def bfl_to_diffusers(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """A BFL / ldm FLUX.2 VAE state on AutoencoderKLFlux2's keys (the tensors shared, the
    mid attention 1x1 conv weights viewed as [C, C])."""
    n_up = up_blocks(state)
    out = {}
    for key, t in state.items():
        out[bfl_key(key, n_up)] = t.reshape(bfl_shape(key, tuple(t.shape)))
    return out


def diffusers_shapes(shapes: dict[str, tuple]) -> dict[str, tuple]:
    """Key -> shape on AutoencoderKLFlux2's keys, from either key style."""
    if not is_bfl(shapes):
        return dict(shapes)
    n_up = up_blocks(shapes)
    return {bfl_key(k, n_up): bfl_shape(k, tuple(s)) for k, s in shapes.items()}


def _count(shapes, pattern: str) -> int:
    n = 0
    while pattern.format(n) in shapes:
        n += 1
    return n


def infer_config(shapes: dict[str, tuple]) -> dict:
    """AutoencoderKLFlux2 init kwargs of a state given as key -> shape (either key style):
    the block widths, layers per block, channels and conv / attention presence from the
    shapes, the rest the stock config (`STOCK`)."""
    s = diffusers_shapes(shapes)
    down = _count(s, "encoder.down_blocks.{}.resnets.0.conv1.weight")
    up = _count(s, "decoder.up_blocks.{}.resnets.0.conv2.weight")
    if down == 0 or up != down:
        raise ValueError(
            f"a FLUX.2 VAE has as many up as down blocks, got {down} / {up}"
        )
    layers = _count(s, "encoder.down_blocks.0.resnets.{}.conv1.weight")
    dec_layers = _count(s, "decoder.up_blocks.0.resnets.{}.conv1.weight")
    if dec_layers != layers + 1:
        raise ValueError(
            f"decoder blocks hold {dec_layers} resnets, expected {layers + 1}"
        )
    latent = s["decoder.conv_in.weight"][1]
    bn = s["bn.running_mean"][0]
    patch = round((bn // latent) ** 0.5)
    widths = [
        s[f"encoder.down_blocks.{i}.resnets.0.conv1.weight"][0] for i in range(down)
    ]
    dec = [
        s[f"decoder.up_blocks.{up - 1 - i}.resnets.0.conv2.weight"][0]
        for i in range(up)
    ]
    return {
        **STOCK,
        "in_channels": s["encoder.conv_in.weight"][1],
        "out_channels": s["decoder.conv_out.weight"][0],
        "down_block_types": ("DownEncoderBlock2D",) * down,
        "up_block_types": ("UpDecoderBlock2D",) * up,
        "block_out_channels": tuple(widths),
        "decoder_block_out_channels": tuple(dec),
        "layers_per_block": layers,
        "latent_channels": latent,
        "use_quant_conv": "quant_conv.weight" in s,
        "use_post_quant_conv": "post_quant_conv.weight" in s,
        "mid_block_add_attention": "encoder.mid_block.attentions.0.to_q.weight" in s,
        "patch_size": (patch, patch),
    }


def read_state(path: str | Path) -> dict[str, Tensor]:
    """Every tensor of a single safetensors file, on the host."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


@torch.no_grad()
def load_vae_file(path: str | Path) -> AutoencoderKLFlux2:
    """AutoencoderKLFlux2 in fp32 from one safetensors file of either key style, its
    config inferred from the shapes; KeyError naming missing / unexpected keys."""
    state = read_state(path)
    if is_bfl(state):
        state = bfl_to_diffusers(state)
    config = infer_config({k: tuple(t.shape) for k, t in state.items()})
    with torch.device("meta"):
        vae = AutoencoderKLFlux2(**config)
    want = set(vae.state_dict())
    missing, unexpected = sorted(want - set(state)), sorted(set(state) - want)
    if missing or unexpected:
        raise KeyError(
            f"{path}: missing {missing[:6]} ({len(missing)}), unexpected "
            f"{unexpected[:6]} ({len(unexpected)})"
        )
    fp32 = {k: t.float() if t.is_floating_point() else t for k, t in state.items()}
    vae.load_state_dict(fp32, strict=True, assign=True)
    left = [n for n, t in (*vae.named_parameters(), *vae.named_buffers()) if t.is_meta]
    if left:
        raise ValueError(f"{path}: tensors left unloaded {left[:6]}")
    return vae.eval()
