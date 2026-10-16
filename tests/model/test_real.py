"""Real-weight parity on one GPU: ours vs the official transformer.

fp32: ours equals the official fp32 forward to fp32 rounding. bf16: ours is no further
from the official fp32 output than the official bf16 forward is.
"""

import gc
import os
from pathlib import Path

import pytest
import torch

from kohakumoss.inference.reference import official_module, rel_error
from kohakumoss.model import load_backbone
from kohakumoss.types.seqinfo import build_seqinfo

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]


def _official(weights, dtype, dev):
    mod = official_module(weights)
    ref = mod.MOSSImageTransformer.from_pretrained(
        str(weights / "transformer"), torch_dtype=dtype
    )
    return ref.to(dev).eval().requires_grad_(False)


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def test_forward_matches_official():
    weights = Path(os.environ["KOHAKUMOSS_WEIGHTS"]) / "base"
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    lat = torch.randn((1, 128, 16, 24), generator=g).to(dev)
    text = torch.randn((20, 4096), generator=g).to(dev)
    sigma = torch.tensor([0.6], device=dev)
    seq = build_seqinfo(torch.tensor([[16, 24]]), torch.tensor([20])).to(dev)
    out = {}
    with torch.inference_mode():
        for dtype in (torch.float32, torch.bfloat16):
            ref = _official(weights, dtype, dev)
            out[f"official_{dtype}"] = ref(
                lat, sigma * 1000, text[None].to(dtype), return_dict=False
            )[0].float()
            del ref
            _free()
            ours, report = load_backbone(weights, device=dev, dtype=dtype)
            assert report.ok
            o = ours(lat[0].flatten(1).T, text.to(dtype), sigma, seq).float()
            out[f"ours_{dtype}"] = o.T.reshape(1, 128, 16, 24)
            del ours
            _free()
    ref32 = out["official_torch.float32"]
    assert rel_error(out["ours_torch.float32"], ref32) < 1e-4
    ours16 = rel_error(out["ours_torch.bfloat16"], ref32)
    official16 = rel_error(out["official_torch.bfloat16"], ref32)
    assert ours16 < 1.25 * official16, (ours16, official16)
