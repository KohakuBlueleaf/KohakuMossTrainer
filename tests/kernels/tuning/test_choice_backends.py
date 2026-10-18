"""On a GPU, the kernel-vs-compile choice builds and times only the arch's most-prior
backend (TileLang on sm70 / sm75, Triton on sm80+) and compile: the other kernel
backend never launches."""

import sys

import pytest
import torch

from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.compile import choice
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

pytestmark = pytest.mark.gpu


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    mode = tuner.get_mode()
    tuner.set_mode("table")
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    choice.reset()
    yield
    tuner.set_mode(mode)
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    choice.reset()


def _count_launches(backend: str, monkeypatch) -> dict:
    """Count kernel launches of `backend` (Triton `Launcher`, TileLang `OnDevice`)."""
    calls = {"n": 0}
    match backend:
        case "triton":
            cls = sys.modules["kohakumoss_kernels.backends.triton._common"].Launcher
            orig = cls._launch

            def launch(self, grid, args, kwargs):
                calls["n"] += 1
                return orig(self, grid, args, kwargs)

            monkeypatch.setattr(cls, "_launch", launch)
        case "tilelang":
            cls = sys.modules[
                "kohakumoss_kernels.backends.tilelang.blocks._blockops"
            ].OnDevice
            orig = cls.__call__

            def call(self, *args):
                calls["n"] += 1
                return orig(self, *args)

            monkeypatch.setattr(cls, "__call__", call)
    return calls


def _inputs(device):
    seq = build_seqinfo(torch.tensor([[8, 8], [4, 6]]), torch.tensor([9, 5]))
    n, d = seq.n_joint, 256
    x = torch.randn(n, d, device=device).half().requires_grad_(True)
    mod = 0.1 * torch.randn(2, 6, d, device=device)
    return (
        x,
        torch.ones(d, device=device),
        mod[:, 0],
        mod[:, 1],
        seq.seg_joint.to(device),
    )


def _other(op) -> str:
    return "tilelang" if op.backend == "triton" else "triton"


def check_other_never_launches(op, device, monkeypatch) -> None:
    other = _other(op)
    calls = _count_launches(other, monkeypatch)
    x, w, shift, scale, seg = _inputs(device)
    op(x, w, shift, scale, seg, 1e-6).float().sum().backward()
    op.decide(x, w, shift, scale, seg, 1e-6)
    assert calls["n"] == 0, f"{other} launched {calls['n']} times under {op.backend}"


def _resolved(device):
    op = registry.resolve("rms_norm_mod", device, torch.float16)
    if op.backend not in ("triton", "tilelang"):
        pytest.skip(f"rms_norm_mod binds {op.backend} here")
    other = _other(op)
    if registry.available("rms_norm_mod", device, torch.float16)[other] is not None:
        pytest.skip(f"{other} cannot serve rms_norm_mod here")
    return op


def test_choice_launches_only_the_most_prior_backend(device, fresh, monkeypatch):
    op = _resolved(device)
    major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 9
    assert op.backend == ("tilelang" if major < 8 else "triton")
    check_other_never_launches(op, device, monkeypatch)


def test_negative_sibling_kernel_launches(device, fresh, monkeypatch):
    """A choice that also times the other backend's kernel must be caught."""
    op = _resolved(device)
    other = registry._STATE.impls["rms_norm_mod"][_other(op)].fn.__wrapped__
    monkeypatch.setattr(choice, "CHOICES", ("kernel", "sibling", "compile"))
    op.fn.impls["sibling"] = other
    try:
        must_fail(
            lambda: check_other_never_launches(op, device, monkeypatch),
            "sibling kernel",
        )
    finally:
        op.fn.impls.pop("sibling", None)
