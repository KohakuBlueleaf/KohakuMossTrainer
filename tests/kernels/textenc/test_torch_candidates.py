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


def _torch_path(mod, device) -> None:
    """Exactly one PyTorch path: torch varlen where it runs, else the SDPA loop."""
    choice = mod.VarlenChoice("test", lambda *a: None, lambda q, s, n: "k")
    impls = choice.impls(device)
    if mod.flash_varlen_usable(device):
        assert impls.get("torch_varlen") is mod.torch_varlen and "sdpa" not in impls
    else:
        assert impls.get("sdpa") is mod.sdpa_loop and "torch_varlen" not in impls


def test_varlen_choice_has_one_torch_path():
    mod = _module("kohakumoss_kernels.backends.varlen_choice")
    for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
        _torch_path(mod, torch.device(device))


def test_varlen_choice_has_one_torch_path_negative(monkeypatch):
    mod = _module("kohakumoss_kernels.backends.varlen_choice")

    def both(self, device):
        return {self.backend: self.kernel, "sdpa": mod.sdpa_loop}

    monkeypatch.setattr(mod, "flash_varlen_usable", lambda device: True)
    monkeypatch.setattr(mod.VarlenChoice, "impls", both)
    must_fail(lambda: _torch_path(mod, torch.device("cpu")), "the syncing SDPA loop")


def test_varlen_variants_change_the_shape_every_call():
    mod = _module("kohakumoss_kernels.backends.varlen_choice")
    q = torch.zeros(300, 2, 8)
    cu = torch.tensor([0, 100, 300], dtype=torch.int32)
    variants = mod._variants(q, cu)
    sizes = [n for n, _ in variants]
    assert len(set(sizes)) == len(sizes) == mod.JITTER
    for n, c in variants:
        assert int(c[-1]) == n and int(c[1]) == 100 and int(c[-1]) > int(c[-2])
    short = mod._variants(q[:103], torch.tensor([0, 100, 103], dtype=torch.int32))
    assert [n for n, _ in short] == [103, 102, 101]


def test_quant_linears_have_pytorch_paths():
    quant = _module("kohakumoss_kernels.quant")
    tri = _module("kohakumoss_kernels.backends.triton.quant")
    cands = tri._candidates(4096)
    assert all(p in cands for p in quant.PYTORCH_PATHS)
