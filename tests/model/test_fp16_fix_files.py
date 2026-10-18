"""fp16-fix checkpoint files without a model: the diffusers writer records the fold on
every shard and the index and leaves the tensors as given; read_fp16_fix reads a dir and
rejects disagreeing shards; the fold's per-family factors and their fp16 values against a
table stated here; the fused single-file layout against the Comfy-Org file's. Models are
the real architecture on the meta device (shapes only)."""

import json
import re

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kohakumoss.model.backbone import MossBackbone
from kohakumoss.model.io.keymap import FP16_FIX_KEY, fuse_qkv
from kohakumoss.model.io.load import read_fp16_fix
from kohakumoss.model.io.save import write_diffusers
from kohakumoss.model.io.source import INDEX
from kohakumoss.model.numerics.fold import _multiply, _row_scales, _scales
from kohakumoss.types.arch import PRESETS
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
CFG = PRESETS["base"]
FIX = {"c": 1 / 16, "s_attn": 0.5, "s_ffn": 0.25}
RECORD = json.dumps({"c": 0.0625, "via": "gate", "s_attn": 0.5, "s_ffn": 0.25})
COMFY_KEYS = 379  # tensors in the Comfy-Org moss_image_base_bf16.safetensors


def constructed_state() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    state = {f"blocks.{i}.weight": torch.randn(64, 8, generator=g) for i in range(6)}
    state["blocks.0.bias"] = torch.randn(64, generator=g).to(torch.float16)
    state["rope.bands"] = torch.rand(16, generator=g)
    return state


def shard_files(d) -> list:
    index = json.loads((d / INDEX).read_text(encoding="utf-8"))
    return [d / name for name in sorted(set(index["weight_map"].values()))]


def assert_fix_dir(d, state: dict, record: str) -> None:
    """Every shard and the index carry `record`; the tensors are `state`'s, unchanged."""
    files = shard_files(d)
    assert len(files) > 1
    index = json.loads((d / INDEX).read_text(encoding="utf-8"))
    assert index["metadata"].get(FP16_FIX_KEY) == record
    seen = {}
    for f in files:
        with safe_open(str(f), "pt") as h:
            meta = h.metadata() or {}
            assert meta.get(FP16_FIX_KEY) == record, f.name
            assert meta.get("format") == "pt", f.name
            seen.update({k: h.get_tensor(k) for k in h.keys()})
    assert sorted(seen) == sorted(state)
    for k, t in state.items():
        assert seen[k].dtype == t.dtype and torch.equal(seen[k], t), k
    assert read_fp16_fix(d) == json.loads(record)


def test_diffusers_writer_records_the_fold_and_keeps_the_tensors(tmp_path):
    state = constructed_state()
    d = tmp_path / "transformer"
    write_diffusers(state, d, CFG, shard_bytes=1000, metadata={FP16_FIX_KEY: RECORD})
    assert_fix_dir(d, state, RECORD)
    assert read_fp16_fix(tmp_path) == json.loads(RECORD)
    unfolded = dict(state, **{"blocks.1.weight": state["blocks.1.weight"] * 16})
    must_fail(lambda: assert_fix_dir(d, unfolded, RECORD), "tensors unfolded on save")
    plain = tmp_path / "plain"
    write_diffusers(state, plain, CFG, shard_bytes=1000)
    must_fail(lambda: assert_fix_dir(plain, state, RECORD), "no fold record")
    assert read_fp16_fix(plain) is None
    with safe_open(str(shard_files(plain)[0]), "pt") as h:
        assert not (h.metadata() or {})


def rejects(d) -> None:
    with pytest.raises(ValueError, match="differs across the files"):
        read_fp16_fix(d)


def test_read_fp16_fix_rejects_disagreeing_shards(tmp_path):
    state = constructed_state()
    d = tmp_path / "transformer"
    write_diffusers(state, d, CFG, shard_bytes=1000, metadata={FP16_FIX_KEY: RECORD})
    must_fail(lambda: rejects(d), "a consistent dir")
    first = shard_files(d)[0]
    with safe_open(str(first), "pt") as h:
        tensors = {k: h.get_tensor(k) for k in h.keys()}
    save_file(tensors, str(first), {"format": "pt"})
    rejects(d)
    other = json.dumps({**json.loads(RECORD), "c": 0.125})
    save_file(tensors, str(first), {"format": "pt", FP16_FIX_KEY: other})
    rejects(d)


def expected_scale(key: str, rows: int, fix: dict) -> torch.Tensor:
    """The fold's per-row factor of one official tensor (docs/concepts/numerics.md)."""
    c, sa, sf = fix["c"], fix["s_attn"], fix["s_ffn"]
    s = torch.ones(rows, dtype=torch.float64)
    writer = (
        key.startswith("video_embedder.")
        or key.startswith("text_embedder.proj.")
        or re.fullmatch(
            r"text_embedder\.blocks\.\d+\.(proj|ffn_down)\.(weight|bias)", key
        )
    )
    if writer:
        s[:] = c
    elif key.startswith("shared_block_modulation."):
        s[2 * CFG.dim : 3 * CFG.dim] = c / sa
        s[5 * CFG.dim : 6 * CFG.dim] = c / sf
    elif re.fullmatch(r"layers\.\d+\.proj\.o\.weight", key):
        s[:] = sa
    elif re.fullmatch(r"layers\.\d+\.proj\.ffn_down\.weight", key):
        s[:] = sf
    return s


def swapped_gates(key: str, rows: int, fix: dict) -> torch.Tensor:
    return expected_scale(
        key, rows, {**fix, "s_attn": fix["s_ffn"], "s_ffn": fix["s_attn"]}
    )


def fold_factors() -> dict[str, torch.Tensor]:
    model = MossBackbone(CFG, device="meta")
    rows = _row_scales(
        model, _scales(model, FIX["c"]), "gate", (FIX["s_attn"], FIX["s_ffn"])
    )
    shapes = {k: v.shape for k, v in model.state_dict().items()}
    out = {}
    for k, shape in shapes.items():
        s = rows.get(k, 1.0)
        n = shape[0] if shape else 1
        out[k] = (
            s.double()
            if isinstance(s, torch.Tensor)
            else torch.full((n,), float(s), dtype=torch.float64)
        )
    return out


def assert_factors(table) -> None:
    for k, s in fold_factors().items():
        assert torch.equal(s, table(k, s.shape[0], FIX)), k


def test_fold_factors_per_family_match_the_stated_table():
    assert_factors(expected_scale)
    must_fail(
        lambda: assert_factors(swapped_gates), "attention / FFN gate rows swapped"
    )


def fp16_reference(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """bf16 `t` times per-row `s` in fp64, rounded once to fp16 (numpy, IEEE RNE)."""
    x = t.double().numpy() * s.numpy().reshape(-1, *([1] * (t.dim() - 1)))
    return torch.from_numpy(x.astype(np.float16))


def assert_values(table) -> None:
    """Fold arithmetic on constructed bf16 tensors (normal, subnormal-bound and zero
    values) then the fp16 cast equals the table factor applied once, bit for bit."""
    g = torch.Generator().manual_seed(1)
    keys = (
        "video_embedder.weight",
        "shared_block_modulation.weight",
        "layers.3.proj.o.weight",
        "layers.3.proj.ffn_down.weight",
        "layers.3.proj.q.weight",
    )
    factors = fold_factors()
    for k in keys:
        rows = factors[k].shape[0]
        mag = torch.exp2(torch.randint(-30, 8, (rows, 4), generator=g).float())
        t = (mag * torch.randn(rows, 4, generator=g)).to(torch.bfloat16)
        t[0, 0] = 0
        got = _multiply(t, factors[k].float(), inverse=False).to(torch.float16)
        want = fp16_reference(t, table(k, rows, FIX))
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), k


def test_folded_fp16_values_match_the_table():
    assert_values(expected_scale)
    must_fail(lambda: assert_values(swapped_gates), "attention / FFN gate rows swapped")


def assert_comfy_layout(state: dict) -> None:
    """The Comfy-Org single-file layout: bare keys, fused q/k/v, rope.bands kept."""
    assert len(state) == COMFY_KEYS
    assert "rope.bands" in state
    for i in range(CFG.depth):
        assert f"layers.{i}.proj.qkv.weight" in state
        assert tuple(state[f"layers.{i}.proj.qkv.weight"].shape) == (
            3 * CFG.dim,
            CFG.dim,
        )
        assert f"layers.{i}.proj.q.weight" not in state


def test_fused_single_file_layout_matches_comfy_org():
    state = MossBackbone(CFG, device="meta").state_dict()
    assert_comfy_layout(fuse_qkv(dict(state), CFG.depth))
    must_fail(lambda: assert_comfy_layout(dict(state)), "split q/k/v")
