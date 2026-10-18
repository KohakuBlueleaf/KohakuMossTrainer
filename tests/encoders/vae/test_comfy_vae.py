"""ComfyUI single-file FLUX.2 VAE (model-free): the BFL / ldm key map against diffusers'
own ldm converter on constructed tensors, the mapped key / shape set against
AutoencoderKLFlux2 built on meta, and the config inferred from shapes against the
values of MOSS-Image's `vae/config.json` (small decoder) and the full FLUX.2 decoder."""

import pytest
import torch
from diffusers import AutoencoderKLFlux2
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

from kohakumoss.encoders.vae.comfy import (
    bfl_to_diffusers,
    diffusers_shapes,
    infer_config,
)
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

ENC = (128, 256, 512, 512)
SMALL_DEC = (96, 192, 384, 384)
LATENT = 32
LAYERS = 2
# MOSS-Image vae/config.json: the AutoencoderKLFlux2 init fields
HF_SMALL = {
    "act_fn": "silu",
    "batch_norm_eps": 0.0001,
    "batch_norm_momentum": 0.1,
    "block_out_channels": (128, 256, 512, 512),
    "decoder_block_out_channels": (96, 192, 384, 384),
    "down_block_types": ("DownEncoderBlock2D",) * 4,
    "in_channels": 3,
    "latent_channels": 32,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "out_channels": 3,
    "patch_size": (2, 2),
    "sample_size": 1024,
    "up_block_types": ("UpDecoderBlock2D",) * 4,
    "use_post_quant_conv": True,
    "use_quant_conv": True,
}
LEFT_TO_US = {
    "quant_conv.weight",
    "quant_conv.bias",
    "post_quant_conv.weight",
    "post_quant_conv.bias",
    "bn.running_mean",
    "bn.running_var",
    "bn.num_batches_tracked",
}


def _conv(out: dict, name: str, cout: int, cin: int, k: int) -> None:
    out[f"{name}.weight"] = (cout, cin, k, k)
    out[f"{name}.bias"] = (cout,)


def _norm(out: dict, name: str, c: int) -> None:
    out[f"{name}.weight"] = (c,)
    out[f"{name}.bias"] = (c,)


def _resnet(out: dict, name: str, cin: int, cout: int) -> None:
    _norm(out, f"{name}.norm1", cin)
    _conv(out, f"{name}.conv1", cout, cin, 3)
    _norm(out, f"{name}.norm2", cout)
    _conv(out, f"{name}.conv2", cout, cout, 3)
    if cin != cout:
        _conv(out, f"{name}.nin_shortcut", cout, cin, 1)


def _mid(out: dict, side: str, c: int) -> None:
    _resnet(out, f"{side}.mid.block_1", c, c)
    _norm(out, f"{side}.mid.attn_1.norm", c)
    for part in ("q", "k", "v", "proj_out"):
        _conv(out, f"{side}.mid.attn_1.{part}", c, c, 1)
    _resnet(out, f"{side}.mid.block_2", c, c)


def bfl_shapes(enc=ENC, dec=SMALL_DEC, layers=LAYERS, latent=LATENT) -> dict:
    """Key -> shape of a FLUX.2 VAE in the BFL / ldm layout (written from the ldm
    Encoder / Decoder definitions, independent of the map under test): decoder level b
    has width dec[b], level 0 the finest."""
    out: dict = {}
    _conv(out, "encoder.conv_in", enc[0], 3, 3)
    cin = enc[0]
    for i, c in enumerate(enc):
        for j in range(layers):
            _resnet(out, f"encoder.down.{i}.block.{j}", cin if j == 0 else c, c)
        cin = c
        if i < len(enc) - 1:
            _conv(out, f"encoder.down.{i}.downsample.conv", c, c, 3)
    _mid(out, "encoder", enc[-1])
    _norm(out, "encoder.norm_out", enc[-1])
    _conv(out, "encoder.conv_out", 2 * latent, enc[-1], 3)
    _conv(out, "encoder.quant_conv", 2 * latent, 2 * latent, 1)
    _conv(out, "decoder.post_quant_conv", latent, latent, 1)
    _conv(out, "decoder.conv_in", dec[-1], latent, 3)
    _mid(out, "decoder", dec[-1])
    for b in reversed(range(len(dec))):
        cin = dec[b + 1] if b < len(dec) - 1 else dec[-1]
        for j in range(layers + 1):
            _resnet(out, f"decoder.up.{b}.block.{j}", cin if j == 0 else dec[b], dec[b])
        if b > 0:
            _conv(out, f"decoder.up.{b}.upsample.conv", dec[b], dec[b], 3)
    _norm(out, "decoder.norm_out", dec[0])
    _conv(out, "decoder.conv_out", 3, dec[0], 3)
    for name in ("running_mean", "running_var"):
        out[f"bn.{name}"] = (4 * latent,)
    out["bn.num_batches_tracked"] = ()
    return out


def _tagged(shapes: dict) -> dict:
    """One small tensor per key holding the key's index (4-D for 4-D weights, so the
    attention 1x1 convs keep their conv rank)."""
    state = {}
    for i, (k, s) in enumerate(sorted(shapes.items())):
        rank = len(s)
        state[k] = torch.full((1,) * rank, float(i)) if rank else torch.tensor(i)
    return state


def check_vs_diffusers(mapper) -> None:
    """`mapper` equals diffusers' ldm converter on every key it maps, and maps exactly
    the keys that converter leaves (`LEFT_TO_US`) on top."""
    state = _tagged(bfl_shapes(dec=ENC))
    blocks = {"down_block_types": ["x"] * 4, "up_block_types": ["x"] * 4}
    theirs = convert_ldm_vae_checkpoint(dict(state), blocks)
    ours = mapper(state)
    assert set(ours) - set(theirs) == LEFT_TO_US, sorted(set(ours) ^ set(theirs))[:8]
    assert set(theirs) <= set(ours)
    for key, t in theirs.items():
        assert torch.equal(ours[key], t), key


def check_vs_meta(mapper_shapes, enc=ENC, dec=SMALL_DEC) -> None:
    """The mapped key / shape set equals AutoencoderKLFlux2 of the inferred config."""
    shapes = mapper_shapes(bfl_shapes(enc, dec))
    with torch.device("meta"):
        vae = AutoencoderKLFlux2(**infer_config(shapes))
    want = {k: tuple(t.shape) for k, t in vae.state_dict().items()}
    assert set(shapes) == set(want), sorted(set(shapes) ^ set(want))[:8]
    for key, s in want.items():
        assert tuple(shapes[key]) == s, (key, shapes[key], s)


def _dropped(state):
    return {k: v for k, v in bfl_to_diffusers(state).items() if ".to_out." not in k}


def _unreversed_key(key: str, n: int = 4) -> str:
    """Undo the up-block reversal of a mapped decoder key."""
    head = "decoder.up_blocks."
    if not key.startswith(head):
        return key
    index, _, rest = key[len(head) :].partition(".")
    return f"{head}{n - 1 - int(index)}.{rest}"


def _unreversed(state):
    return {_unreversed_key(k): v for k, v in bfl_to_diffusers(state).items()}


def _unreversed_shapes(shapes):
    return {_unreversed_key(k): v for k, v in diffusers_shapes(shapes).items()}


def test_bfl_map_matches_the_diffusers_ldm_converter():
    check_vs_diffusers(bfl_to_diffusers)
    must_fail(lambda: check_vs_diffusers(_dropped), "a dropped mapping entry")
    must_fail(lambda: check_vs_diffusers(_unreversed), "up blocks not reversed")


@pytest.mark.parametrize("dec", (SMALL_DEC, ENC), ids=("small", "full"))
def test_mapped_keys_build_the_diffusers_vae(dec):
    check_vs_meta(diffusers_shapes, ENC, dec)
    must_fail(lambda: check_vs_meta(_unreversed_shapes, ENC, SMALL_DEC), "unreversed")


def check_config(config: dict, want: dict) -> None:
    for key, value in want.items():
        got = config[key]
        got = tuple(got) if isinstance(got, list) else got
        assert got == value, (key, got, value)


def test_config_from_shapes_of_either_key_style():
    small = infer_config(bfl_shapes())
    check_config(small, HF_SMALL)
    with torch.device("meta"):
        vae = AutoencoderKLFlux2(**HF_SMALL)
    native = infer_config({k: tuple(t.shape) for k, t in vae.state_dict().items()})
    check_config(native, HF_SMALL)
    full = infer_config(bfl_shapes(dec=ENC))
    assert tuple(full["decoder_block_out_channels"]) == ENC
    assert max(full["decoder_block_out_channels"]) == 512
    must_fail(lambda: check_config(full, HF_SMALL), "the full decoder read as small")
    assert len(bfl_shapes(dec=ENC)) == 251
