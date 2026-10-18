"""ComfyUI's fused q/k/v layout: the fused LoRA and the fused diff carry each layer's
dW in its rows, and the split gives back each layer's dW (checked against the dW the
unfused sets compute, not against a round trip)."""

import pytest
import torch

from kohakumoss.adapters.export.fused import (
    fused_diff,
    fused_lora,
    is_lora,
    qkv_groups,
    split_qkv,
)
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
OUT, IN = 12, 10


def lora(rank: int, alpha: float, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "lora_up.weight": torch.randn(OUT, rank, generator=g),
        "lora_down.weight": torch.randn(rank, IN, generator=g),
        "alpha": torch.tensor(alpha),
    }


def lora_dw(t: dict) -> torch.Tensor:
    up, down = t["lora_up.weight"], t["lora_down.weight"]
    scale = float(t["alpha"]) / down.shape[0] if "alpha" in t else 1.0
    return up @ down * scale


PARTS = {"q": lora(2, 1.0, 0), "k": lora(3, 6.0, 1), "v": lora(4, 4.0, 2)}
ROWS = {"q": OUT, "k": OUT, "v": OUT}


def expected(parts: dict) -> torch.Tensor:
    return torch.cat(
        [lora_dw(parts[n]) if n in parts else torch.zeros(OUT, IN) for n in "qkv"]
    )


def assert_close(a: torch.Tensor, b: torch.Tensor) -> None:
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_fused_lora_carries_each_layer():
    fused = fused_lora(PARTS, ROWS)
    assert is_lora(fused) and fused["lora_down.weight"].shape[0] == 9
    assert_close(lora_dw(fused), expected(PARTS))
    no_k = {"q": PARTS["q"], "v": PARTS["v"]}
    assert_close(lora_dw(fused_lora(no_k, ROWS)), expected(no_k))
    broken = {**fused, "alpha": torch.tensor(1.0)}
    must_fail(lambda: assert_close(lora_dw(broken), expected(PARTS)), "alpha lost")


def test_fused_diff_stacks_rows():
    diffs = {n: {"diff": lora_dw(p)} for n, p in PARTS.items() if n != "k"}
    fused = fused_diff(diffs, ROWS, IN)
    assert_close(fused["diff"], expected({"q": PARTS["q"], "v": PARTS["v"]}))
    swapped = {"q": diffs["v"], "v": diffs["q"]}
    must_fail(
        lambda: assert_close(fused_diff(swapped, ROWS, IN)["diff"], fused["diff"]),
        "rows out of q;k;v order",
    )


def test_split_gives_each_layer_back():
    fused = fused_lora(PARTS, ROWS)
    stem = "lora_unet_layers_3_proj_qkv"
    keys = {f"{stem}.{k}": v for k, v in fused.items()}
    keys["lora_unet_layers_3_proj_o.diff"] = torch.ones(OUT, IN)
    out = split_qkv(keys)
    assert "lora_unet_layers_3_proj_o.diff" in out
    for name, part in PARTS.items():
        got = {
            k.split(".", 1)[1]: v
            for k, v in out.items()
            if k.startswith(f"lora_unet_layers_3_proj_{name}.")
        }
        assert got["lora_down.weight"].shape[0] == part["lora_down.weight"].shape[0]
        assert_close(lora_dw(got), lora_dw(part))
    diff = {"diffusion_model.layers.0.proj.qkv.diff": expected(PARTS)}
    split = split_qkv(diff)
    assert_close(split["diffusion_model.layers.0.proj.k.diff"], lora_dw(PARTS["k"]))
    with pytest.raises(KeyError):
        split_qkv({f"{stem}.hada_w1_a": torch.ones(1)})


def test_groups_find_joint_attention_only():
    paths = [
        "layers.0.proj.q",
        "layers.0.proj.v",
        "layers.0.proj.o",
        "text_embedder.blocks.0.qkv",
    ]
    assert qkv_groups(paths) == {
        "layers.0.proj.qkv": {"q": "layers.0.proj.q", "v": "layers.0.proj.v"}
    }
