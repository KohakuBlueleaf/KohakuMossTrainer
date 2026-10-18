"""Batched text encoding on the real Qwen3.5 weights: every layout of the kernel forward
against transformers' own encode at the same precision, both measured against an fp64
encode (the kernel forward on the host, whose ops are checked against independent
references in tests/kernels). A layout fails only when its error statistics regress
against transformers' (tests/_util/error_stats.py). Each dtype also records the
planner's resident footprint (config only) beside the real encoder's measured device
allocation. See docs/internals/text-encoder.md.
"""

import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from kohakumoss.encoders.text import QwenTextEncoder, text_plan, tokenizer_dir
from kohakumoss.encoders.text.forward import make_pack
from kohakumoss.training.build.text_estimate import planned_text_sizes
from tests._util.negative import must_fail
from tests._util.textenc_ref import assert_encode_no_regression, encode_error

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]

SUBJECTS = (
    "a cat", "an old lighthouse on a cliff", "a girl with long silver hair",
    "a busy night market", "a red sports car", "a bowl of ramen",
    "a fox in the snow", "an astronaut on mars",
)  # fmt: skip
STYLES = (
    "oil painting", "watercolor", "35mm photo, shallow depth of field",
    "anime screencap", "pencil sketch", "3d render, octane",
)  # fmt: skip
TAGS = (
    "masterpiece", "best quality", "looking at viewer", "smile", "outdoors",
    "rain", "neon lights", "reflections", "dramatic lighting", "wide shot",
)  # fmt: skip


def prompt_set() -> list[str]:
    """32 captions, 1 to ~300 tokens: tag lists, sentences, repeats and the empty one."""
    out = [""]
    for i in range(31):
        subject, style = SUBJECTS[i % len(SUBJECTS)], STYLES[i % len(STYLES)]
        tags = ", ".join(TAGS[: 1 + i % len(TAGS)])
        match i % 3:
            case 0:
                out.append(f"{subject}, {tags}")
            case 1:
                out.append(f"A {style} of {subject}. " * (1 + i // 6))
            case _:
                out.append(f"{style}, {subject}, {tags}, " * (1 + i // 8))
    return out


PROMPTS = prompt_set()
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
LAYOUTS = {
    "packed": {"layout": "packed"},
    "padded": {"layout": "padded"},
    "kernels_single": {"layout": "single"},
}
HF = {
    "single": {"forward": "hf", "layout": "single"},
    "padded": {"forward": "hf", "layout": "padded"},
}


def _root() -> Path:
    root = Path(os.environ["KOHAKUMOSS_WEIGHTS"])
    for cand in (root / "base", root):
        if (cand / "text_encoder").is_dir():
            return cand / "text_encoder"
    pytest.skip("no text_encoder/ under KOHAKUMOSS_WEIGHTS")


@pytest.fixture(scope="module")
def ref64():
    path = _root()
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir(path)))
    model = AutoModel.from_pretrained(str(path), dtype=torch.float64)
    te = QwenTextEncoder(model, tok, "cpu", {"memory": "cpu"}, output_device="cpu")
    out = te.encode(PROMPTS)
    te.release()
    return tok, out


LOADED: dict[str, int] = {}  # dtype -> device bytes the last resident encoder took


def _encoder(tok, dtype: str, spec: dict) -> QwenTextEncoder:
    model = AutoModel.from_pretrained(str(_root()), dtype=DTYPES[dtype])
    plan = text_plan({"memory": "resident", "dtype": dtype, **spec})
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    te = QwenTextEncoder(model, tok, "cuda", plan, output_device="cpu")
    torch.cuda.synchronize()
    LOADED[dtype] = torch.cuda.memory_allocated() - before
    return te


def _record_footprint(record_property, dtype: str) -> None:
    """The planner's resident text-encoder bytes (config only) next to the measured
    allocation of the real encoder, and their relative gap."""
    plan = text_plan({"memory": "resident", "dtype": dtype})
    planned = planned_text_sizes(_root(), plan, torch.device("cuda")).total
    measured = LOADED[dtype]
    record_property(f"{dtype}_te_planned_gib", planned / 2**30)
    record_property(f"{dtype}_te_measured_gib", measured / 2**30)
    record_property(f"{dtype}_te_gap", (planned - measured) / measured)


def _encode(tok, dtype, spec) -> list[torch.Tensor]:
    te = _encoder(tok, dtype, spec)
    out = te.encode(PROMPTS)
    te.release()
    return out


def _record(record_property, key: str, out, ref) -> None:
    for stat, value in encode_error(out, ref).items():
        record_property(f"{key}_{stat}", value)


@pytest.mark.parametrize("dtype", tuple(DTYPES))
def test_layouts_do_not_regress_against_transformers(ref64, dtype, record_property):
    """Each layout (packed, padded, one prompt per call) against transformers in the same
    batching (its unbatched encode for one prompt per call, its padded batch for the
    batched layouts): fails only on a regression vs transformers against fp64. Every
    stats table is recorded."""
    tok, ref = ref64
    hf = {name: _encode(tok, dtype, spec) for name, spec in HF.items()}
    for name, out in hf.items():
        _record(record_property, f"{dtype}_hf_{name}", out, ref)
    outs = {name: _encode(tok, dtype, spec) for name, spec in LAYOUTS.items()}
    for name, out in outs.items():
        _record(record_property, f"{dtype}_{name}", out, ref)
    _record_footprint(record_property, dtype)
    for name, out in outs.items():
        base = hf["single" if LAYOUTS[name]["layout"] == "single" else "padded"]
        assert_encode_no_regression(out, base, ref, f"{dtype} {name}")


@pytest.mark.parametrize("dtype", ("bf16",))
def test_boundary_crossing_pack_fails(ref64, dtype, record_property):
    """The packed forward with one merged segment (positions still reset per prompt)
    must fail the same check."""
    tok, ref = ref64
    unbatched = _encode(tok, dtype, HF["single"])
    te = _encoder(tok, dtype, {"layout": "packed", "batch": len(PROMPTS)})
    rows = te.token_ids(PROMPTS)
    pack = make_pack(rows, te.device, te.pad_id, padded=False)
    pack.cu = torch.tensor([0, pack.ids.numel()], dtype=torch.int32, device=te.device)
    pack.max_len = pack.ids.numel()
    te.placement.begin()
    merged = list(te.forward([pack])[0].to("cpu").split(pack.lengths))
    te.placement.end()
    te.release()
    _record(record_property, f"{dtype}_merged", merged, ref)
    must_fail(
        lambda: assert_encode_no_regression(merged, unbatched, ref, "merged"),
        "prompts attend / recur across their boundaries",
    )
