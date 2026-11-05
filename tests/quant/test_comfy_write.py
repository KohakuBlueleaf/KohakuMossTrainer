"""ComfyUI int8 ConvRot writer (model-free): the codes and scales against the independent
numpy ConvRot formulas (tests/_util/comfy_ref.py), the record bytes against the
Comfy-Org file's record, and the written state read back by the ComfyUI layer parser.
A Sylvester Hadamard, a missing rotation and a wrong record fail."""

import numpy as np
import pytest
import torch

from kohakumoss.quant.comfy_format import comfy_layers, layer_base, read_record
from kohakumoss.quant.comfy_write import convrot_int8, int8_convrot_state
from tests._util.comfy_ref import hadamard_np, unrotate_np
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

COMFY_ORG_RECORD = (
    b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
)
COMFY_ORG_HINT = b'{"attention": "comfy_kitchen_int8"}'


def _weight(n=64, k=512, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn((n, k), generator=g) * 0.02
    w[3, 7] = 0.9
    return w


def _np_codes(
    w: np.ndarray, group: int, h: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rot = (w.astype(np.float64).reshape(-1, group) @ h).reshape(w.shape)
    scale = np.abs(rot).max(axis=1) / 127.0
    return np.clip(np.rint(rot / scale[:, None]), -127, 127), scale


def _check_codes(q, scale, w, group, h) -> None:
    ref_q, ref_s = _np_codes(w.numpy(), group, h)
    np.testing.assert_allclose(scale.numpy()[:, 0], ref_s, rtol=1e-6)
    diff = np.abs(q.numpy().astype(np.float64) - ref_q)
    assert diff.max() <= 1 and (diff > 0).mean() < 1e-3, (diff.max(), (diff > 0).mean())


@pytest.mark.parametrize("group", (16, 64, 256))
def test_convrot_codes_match_numpy(group):
    w = _weight()
    q, scale = convrot_int8(w, group)
    assert q.dtype == torch.int8 and scale.shape == (w.shape[0], 1)
    _check_codes(q, scale, w, group, hadamard_np(group))
    deq = unrotate_np(q.numpy().astype(np.float64) * scale.double().numpy(), group)
    err = np.linalg.norm(deq - w.double().numpy()) / np.linalg.norm(w.double().numpy())
    assert err < 2e-2, err
    plain_s = (w.abs().amax(1) / 127)[:, None]
    plain = torch.round(w / plain_s).to(torch.int8)
    must_fail(
        lambda: _check_codes(plain, plain_s, w, group, hadamard_np(group)),
        "no rotation",
    )
    syl = np.array([[1.0]])
    while syl.shape[0] < group:
        syl = np.block([[syl, syl], [syl, -syl]])
    syl = syl / np.sqrt(group)
    must_fail(lambda: _check_codes(q, scale, w, group, syl), "Sylvester Hadamard")


def _state() -> dict[str, torch.Tensor]:
    return {
        "layers.0.proj.qkv.weight": _weight(96, 256, 1),
        "layers.0.proj.ffn_down.weight": _weight(32, 512, 2),
        "layers.0.norm1.weight": torch.ones(256),
        "layers.1.proj.ffn_down.weight": _weight(32, 512, 3),
        "time_embedding.2.weight": _weight(32, 256, 4),
        "time_embedding.2.bias": torch.zeros(32),
        "rope.bands": torch.arange(4, dtype=torch.float32),
    }


def test_state_layout_and_read_back():
    src = _state()
    out, errors = int8_convrot_state(
        src, r"layers\.\d+\.proj\.[a-z_]+", 256, torch.bfloat16,
        attention_hint="comfy_kitchen_int8",
    )  # fmt: skip
    assert set(errors) == {
        "layers.0.proj.qkv",
        "layers.0.proj.ffn_down",
        "layers.1.proj.ffn_down",
    }
    assert bytes(out["layers.0.proj.qkv.comfy_quant"].tolist()) == COMFY_ORG_RECORD
    assert bytes(out["layers.1.comfy_attention.config"].tolist()) == COMFY_ORG_HINT
    assert "layers.2.comfy_attention.config" not in out
    assert out["time_embedding.2.weight"].dtype == torch.bfloat16
    assert out["layers.0.norm1.weight"].dtype == torch.bfloat16
    assert out["rope.bands"].dtype == torch.float32
    assert out["layers.0.proj.qkv.weight_scale"].dtype == torch.float32
    keys = list(out)
    layers = comfy_layers(keys, out.__getitem__, lambda k: out[k].dtype)
    assert set(layers) == set(errors)
    assert (
        read_record(out["layers.0.proj.ffn_down.comfy_quant"])["convrot_groupsize"]
        == 256
    )
    for path, layer in layers.items():
        got = layer_base(layer, out.__getitem__).weight(torch.float32)
        ref = src[f"{path}.weight"]
        rel = ((got - ref).norm() / ref.norm()).item()
        assert rel == pytest.approx(errors[path], rel=1e-3, abs=1e-6) and rel < 2e-2
    bad = dict(out)
    bad["layers.0.proj.qkv.comfy_quant"] = out["layers.0.proj.qkv.comfy_quant"][:-1]
    must_fail(
        lambda: _assert_record(bad["layers.0.proj.qkv.comfy_quant"]), "truncated record"
    )


def _assert_record(t: torch.Tensor) -> None:
    assert bytes(t.tolist()) == COMFY_ORG_RECORD
