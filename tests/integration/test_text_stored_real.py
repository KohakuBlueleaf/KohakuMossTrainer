"""Pre-quantized text-encoder file on the real Qwen3.5 encoder: an exported file loads
without quantizing and encodes exactly like the quantize-at-load encoder."""

from pathlib import Path

import pytest
import torch
from kohakuengine import run

from kohakumoss.encoders.text.encoder import QwenTextEncoder
from kohakumoss.quant.convert import QuantLinear
from kohakumoss.quant.stored_text import read_text_record, text_of
from kohakumoss_kernels.hardware.arch import arch_of

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ["a cat on a sofa", "a lighthouse, rocky coast, sunset, outdoors"]


def _encode(weights: str, plan: dict) -> list[torch.Tensor]:
    te = QwenTextEncoder.from_pretrained(weights, "cuda", plan)
    return [t.float().cpu() for t in te.encode(PROMPTS)]


@pytest.mark.parametrize("precision", ["fp8", "fp8_compute"])
def test_stored_file_encodes_like_load_time_quantization(
    weights_root, tmp_path, precision
):
    weights = str(weights_root / "base")
    if precision == "fp8_compute" and not arch_of("cuda").has_fp8:
        pytest.skip(
            f"fp8 compute needs a card with fp8; {arch_of('cuda').name} has none"
        )
    out = tmp_path / "te.safetensors"
    run(
        str(ROOT / "scripts/export/text_quant.py"),
        set_overrides={"WEIGHTS": weights, "PRECISION": precision, "OUT": str(out)},
    )
    record = read_text_record(out)
    assert record["precision"] == precision and record["layers"]
    base = {"precision": precision, "memory": "resident", "dtype": "auto"}
    want = _encode(weights, base)
    te = QwenTextEncoder.from_pretrained(weights, "cuda", {**base, "stored": str(out)})
    stored = [m for m in text_of(te.model).modules() if isinstance(m, QuantLinear)]
    assert len(stored) == len(record["layers"])
    got = [t.float().cpu() for t in te.encode(PROMPTS)]
    for a, b in zip(got, want):
        assert torch.equal(a, b), "the stored file encodes differently"
    del te
    dense = _encode(weights, {**base, "precision": "compute"})
    assert not all(
        torch.equal(a, b) for a, b in zip(dense, want)
    ), "the comparison cannot tell quantized from dense"
