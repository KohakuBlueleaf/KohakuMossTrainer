"""Every tuned kernel family times a PyTorch path inside its shortlist, so a kernel is
bound only where it beats PyTorch (the candidate lists, no launches)."""

import importlib

import pytest
import torch

from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"{name}: {err}")


def check_torch_candidate(candidates: list[dict], shortlist: int, impl: str) -> None:
    timed = candidates[:shortlist]
    assert any(c.get("impl") == impl for c in timed), f"no {impl!r} in the shortlist"


def test_triton_causal_attention_has_torch_candidate():
    mod = _module("kohakumoss_kernels.backends.triton.textenc.attention")
    cands = mod.candidates(torch.device("cpu"), torch.float16, 128)
    check_torch_candidate(cands, 12, "sdpa")
    kernels = [c for c in cands if c.get("impl") != "sdpa"]
    must_fail(lambda: check_torch_candidate(kernels, 12, "sdpa"), "no sdpa candidate")


def test_triton_delta_rule_has_torch_candidate():
    mod = _module("kohakumoss_kernels.backends.triton.textenc.delta")
    cands = mod.candidates(torch.device("cpu"), torch.float16, 128)
    check_torch_candidate(cands, mod.SHORTLIST, "torch")
    kernels = [c for c in cands if c is not mod.TORCH]
    must_fail(
        lambda: check_torch_candidate(kernels, mod.SHORTLIST, "torch"),
        "no torch candidate",
    )


def test_tilelang_text_ops_have_torch_candidates():
    mod = _module("kohakumoss_kernels.backends.tilelang.textenc.ops")
    check_torch_candidate(mod.delta_candidates(128), 10, "torch")
    check_torch_candidate(mod.attn_candidates(128), 8, "sdpa")
    must_fail(
        lambda: check_torch_candidate(mod.attn_candidates(128)[2:], 8, "sdpa"),
        "sdpa cut from the shortlist",
    )


def test_tilelang_attention_choice_has_torch_candidate():
    mod = _module("kohakumoss_kernels.backends.tilelang.attention.entry")
    impls = mod.CHOICE.impls(torch.device("cpu"))
    assert "tilelang" in impls and "sdpa" in impls


def test_varlen_choice_always_times_the_sdpa_reference():
    mod = _module("kohakumoss_kernels.backends.varlen_choice")
    choice = mod.VarlenChoice("test", lambda *a: None, lambda q, s, n: "k")
    for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
        impls = choice.impls(torch.device(device))
        assert impls["sdpa"] is mod.sdpa_loop, f"no sdpa reference on {device}"


def test_quant_linears_have_pytorch_paths():
    quant = _module("kohakumoss_kernels.quant")
    tri = _module("kohakumoss_kernels.backends.triton.quant")
    cands = tri._candidates(4096)
    assert all(p in cands for p in quant.PYTORCH_PATHS)
