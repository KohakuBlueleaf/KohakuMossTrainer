"""Selective checkpointing keeps the outputs of the kernel attention and fused-linear
forward custom ops (they are not aten matmuls the policy would otherwise see)."""

import importlib.util

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy

from kohakumoss.memory import runners
from kohakumoss_kernels import registry, saved_ops
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
FORWARDS = {
    "triton": (
        "attn_varlen_triton",
        "triton_linear_situ_fwd",
        "triton_linear_gres_fwd",
        "triton_linear_qk_fwd",
    ),
    "tilelang": (
        "attn_varlen_tilelang",
        "tilelang_linear_situ_fwd",
        "tilelang_linear_gres_fwd",
        "tilelang_linear_qk_fwd",
    ),
}


def _forwards() -> list:
    registry.load_backends()
    ops = []
    for backend, names in FORWARDS.items():
        if importlib.util.find_spec(backend) is None:
            continue
        space = torch.ops.kohakumoss
        ops += [getattr(space, n).default for n in names if hasattr(space, n)]
    return ops


def check_kept(ops) -> None:
    assert ops, "no kernel backend registered its forwards here"
    for op in ops:
        policy = runners._save_matmuls(None, op)
        assert policy == CheckpointPolicy.MUST_SAVE, f"{op} recomputed"
    elementwise = runners._save_matmuls(None, torch.ops.aten.mul.Tensor)
    assert elementwise == CheckpointPolicy.PREFER_RECOMPUTE


def test_kernel_forwards_are_kept():
    ops = _forwards()
    if not ops:
        pytest.skip("neither triton nor tilelang is installed")
    check_kept(ops)


def test_negative_empty_kept(monkeypatch):
    ops = _forwards()
    if not ops:
        pytest.skip("neither triton nor tilelang is installed")
    monkeypatch.setattr(runners, "KEPT", set())
    must_fail(lambda: check_kept(ops), "kernel forwards recomputed")


def test_keep_named_skips_unknown_ops():
    before = set(saved_ops.KEPT)
    saved_ops.keep_named("kohakumoss", "no_such_op_registered")
    assert saved_ops.KEPT == before
