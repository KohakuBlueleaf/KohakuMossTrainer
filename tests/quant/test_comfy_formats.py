"""ComfyUI quantized formats (model-free): the ConvRot int8 base, the W4A8 base (4-bit
and 6-bit codes, codebook and uniform levels, a correction term), the int8 embedding
table and the `comfy_quant` record parsing, each witnessed against the independent
numpy formulas of comfy-kitchen's eager backend (tests/_util/comfy_ref.py) on
constructed weights. Wrong Hadamards, swapped nibbles and a missing rotation fail."""

import hashlib
import json

import numpy as np
import pytest
import torch

from kohakumoss.quant.comfy import split_quantized
from kohakumoss.quant.comfy_format import (
    comfy_layers,
    layer_base,
    read_record,
    stored_format,
)
from kohakumoss.quant.embedding import QuantEmbedding
from kohakumoss.quant.w4a8 import W4A8Weight, decode_codes, unpack_codes
from kohakumoss.quant.weights import QuantWeight
from tests._util.comfy_ref import LUT, decode_np, pack_np, unrotate_np, w4a8_weight_np
from tests._util.hadamard_ref import h64
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _int8_layer(n=96, k=512, seed=0):
    g = _gen(seed)
    q = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8)
    scale = torch.rand(n, generator=g) * 1e-2 + 1e-3
    return q, scale


def _sha(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def _close(got: torch.Tensor, ref: np.ndarray, tol: float = 1e-6) -> None:
    err = np.abs(got.double().numpy() - ref).max() / np.abs(ref).max()
    assert err < tol, err


@pytest.mark.parametrize("group", (16, 64, 256))
def test_convrot_int8_weight_is_the_unrotated_dequant(group):
    q, scale = _int8_layer()
    w = QuantWeight(q, scale, "int8", group).weight(torch.float32)
    ref = unrotate_np(q.numpy().astype(np.float64) * scale.double().numpy()[:, None], group)  # fmt: skip
    _close(w, ref)
    plain = QuantWeight(q, scale, "int8").weight(torch.float32)
    must_fail(lambda: _close(plain, ref), "the rotation left in")

    def sylvester():
        w_rot = q.double() * scale.double()[:, None]
        wrong = (w_rot.reshape(-1, group) @ h64(group, sylvester=True)).reshape(q.shape)
        _close(wrong, ref)

    must_fail(sylvester, "the Sylvester Hadamard")


def test_convrot_linear_equals_the_dense_unrotated_linear():
    q, scale = _int8_layer(64, 256, seed=1)
    base = QuantWeight(q, scale, "int8", 256)
    base.bind(torch.float32)
    x = torch.randn(9, 256, generator=_gen(2))
    w = unrotate_np(q.numpy().astype(np.float64) * scale.double().numpy()[:, None], 256)
    ref = x.double().numpy() @ w.T
    _close(base.linear(x, None), ref, 1e-5)
    unrotated = QuantWeight(q, scale, "int8")
    unrotated.bind(torch.float32)
    must_fail(lambda: _close(unrotated.linear(x, None), ref, 1e-5), "no rotation")


def _codes(n, k, bits, seed):
    return torch.randint(0, 1 << bits, (n, k), generator=_gen(seed)).numpy()


def _s_rel(n, groups, seed):
    s = torch.rand(n, groups, generator=_gen(seed)) * 100 + 8
    return s.to(torch.float8_e4m3fn)


@pytest.mark.parametrize("bits", (4, 6))
@pytest.mark.parametrize("codebook", (True, False))
def test_w4a8_decode_bytes_match_the_reference(bits, codebook):
    n, k, group = 48, 512, 16
    codes = _codes(n, k, bits, seed=bits)
    packed = torch.from_numpy(pack_np(codes, bits))
    s_rel = _s_rel(n, k // group, seed=3)
    book = torch.from_numpy(LUT) if codebook and bits == 4 else None
    got = decode_codes(packed, s_rel, book, group, bits)
    ref = decode_np(packed.numpy(), s_rel.float().numpy(), None if book is None else LUT, group, bits)  # fmt: skip
    assert _sha(got.numpy()) == _sha(ref)
    assert np.array_equal(unpack_codes(packed, k, bits).numpy(), codes)

    def swapped():
        flipped = (packed.view(torch.uint8) >> 4) | (packed.view(torch.uint8) << 4)
        bad = decode_codes(flipped.view(torch.int8), s_rel, book, group, bits)
        assert _sha(bad.numpy()) == _sha(ref)

    must_fail(swapped, "nibbles swapped")


def _w4a8(n=64, k=512, rotation=256, correction=False, seed=5):
    codes = _codes(n, k, 4, seed)
    packed = torch.from_numpy(pack_np(codes, 4))
    s_rel = _s_rel(n, k // 16, seed + 1)
    s_channel = torch.rand(n, generator=_gen(seed + 2)) * 1e-3 + 1e-4
    corr = None
    if correction:
        corr = (torch.randn(k // 16, n, generator=_gen(seed + 3)) * 1e-3).bfloat16()
    w = W4A8Weight(packed, s_rel, s_channel, torch.from_numpy(LUT), 16, rotation, corr)
    grid = decode_np(packed.numpy(), s_rel.float().numpy(), LUT, 16, 4)
    ref = w4a8_weight_np(
        grid,
        s_channel.numpy(),
        16,
        rotation,
        None if corr is None else corr.float().numpy(),
    )
    return w, ref


@pytest.mark.parametrize("correction", (False, True))
def test_w4a8_weight_matches_the_reference(correction):
    w, ref = _w4a8(correction=correction)
    _close(w.weight(torch.float32), ref)
    if not correction:
        int8 = w.to_int8()
        assert int8.rotation == 256 and int8.fmt == "int8"
        _close(int8.weight(torch.float32), ref)
    else:
        with pytest.raises(ValueError):
            w.to_int8()
    plain, _ = _w4a8(rotation=0, correction=correction)
    must_fail(lambda: _close(plain.weight(torch.float32), ref), "the rotation left in")


def test_w4a8_packed_linear_equals_the_int8_linear():
    w, ref = _w4a8(seed=11)
    w.bind(torch.float32)
    int8 = w.to_int8()
    int8.bind(torch.float32)
    x = torch.randn(7, 512, generator=_gen(12))
    packed_y = w.linear(x, None)
    assert torch.equal(packed_y, int8.linear(x, None))
    _close(packed_y, x.double().numpy() @ ref.T, 1e-5)
    assert w.nbytes() < int8.nbytes() < 2 * w.nbytes() + int8.scale.nbytes


def test_int8_embedding_rows_match_the_reference():
    q, scale = _int8_layer(40, 256, seed=7)
    emb = QuantEmbedding(q, scale, 256, torch.float32)
    ids = torch.tensor([[3, 0, 39], [7, 7, 1]])
    full = unrotate_np(q.numpy().astype(np.float64) * scale.double().numpy()[:, None], 256)  # fmt: skip
    _close(emb(ids), full[ids.numpy()])
    assert torch.equal(emb.lookup(ids, torch.device("cpu")), emb(ids))
    _close(emb.weight(torch.float32), full)
    flat = QuantEmbedding(q, scale, 0, torch.float32)
    must_fail(lambda: _close(flat(ids), full[ids.numpy()]), "the rows left rotated")


def _record(payload: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(payload).encode()), dtype=torch.uint8)


def test_comfy_records_parse_into_layers():
    q, scale = _int8_layer(32, 256, seed=9)
    w4 = _w4a8(32, 256, seed=13)[0]
    state = {
        "a.weight": q,
        "a.weight_scale": scale[:, None],
        "a.comfy_quant": _record(
            {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
        ),
        "b.weight": w4.q,
        "b.weight_s_rel": w4.s_rel,
        "b.weight_s_channel": w4.s_channel,
        "b.weight_codebook": w4.codebook,
        "b.comfy_quant": _record(
            {
                "format": "asym_w4a8_int8",
                "group_size": 16,
                "convrot": True,
                "convrot_groupsize": 256,
            }
        ),
        "c.weight": q.to(torch.float8_e4m3fn),
        "c.scale_weight": torch.tensor(0.5),
        "d.weight": torch.randn(4, 4),
    }
    dtypes = {key: t.dtype for key, t in state.items()}
    layers = comfy_layers(list(state), state.__getitem__, dtypes.get)
    assert {p: (lay.fmt, lay.rotation, lay.group) for p, lay in layers.items()} == {
        "a": ("int8", 256, 0),
        "b": ("w4a8", 256, 16),
        "c": ("fp8", 0, 0),
    }
    assert layers["b"].layout == "w4a8-g16-convrot256"
    assert set(layers["a"].keys) == {"a.weight", "a.weight_scale", "a.comfy_quant"}
    assert read_record(state["a.comfy_quant"])["convrot_groupsize"] == 256
    base = layer_base(layers["a"], state.__getitem__)
    assert isinstance(base, QuantWeight) and base.rotation == 256
    assert torch.equal(base.scale, scale)
    assert isinstance(layer_base(layers["b"], state.__getitem__), W4A8Weight)
    assert stored_format(layers) == "mixed"
    assert stored_format({"a": layers["a"]}) == "int8"
    quant, dense = split_quantized(state)
    assert set(quant) == {"a", "b", "c"} and set(dense) == {"d.weight"}
    assert quant["c"].scale.shape == (32,)
    state["e.comfy_quant"] = _record({"format": "nvfp4"})
    with pytest.raises(ValueError, match="nvfp4"):
        comfy_layers(list(state), state.__getitem__, dtypes.get)
    must_fail(
        lambda: _assert_rotation(layer_base(layers["c"], state.__getitem__)),
        "a record-less layer has no rotation",
    )


def _assert_rotation(base) -> None:
    assert base.rotation == 256
