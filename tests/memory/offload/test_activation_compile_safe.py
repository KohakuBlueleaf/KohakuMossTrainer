"""The activation-offload hooks run outside Dynamo: a compiled op whose frame the
backend skips (inductor's bf16 skip on a GPU without native bf16) runs its ops eagerly
under Dynamo's frame callback, and the saved-tensor hooks autograd calls from them still
pack and unpack as plain Python; hooks Dynamo may trace fail on their symbolic sizes."""

import pytest
import torch

from kohakumoss.offload.activation import ActivationOffload
from kohakumoss_kernels.backends.compile.ops import CompiledOp
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
D = 256
TOKENS = (164, 180, 317, 401)


def skips_frame(gm, example_inputs):
    """A backend that gives the frame up, as inductor does for an unsupported dtype."""
    raise torch._dynamo.exc.SkipFrame("BF16 is not supported")


class Traceable(ActivationOffload):
    """The engine with hooks Dynamo may trace (the undecorated methods)."""

    pack = ActivationOffload.pack.__wrapped__
    unpack = ActivationOffload.unpack.__wrapped__


def offloads(engine: ActivationOffload) -> None:
    """rms_norm_mod compiled with dynamic shapes over four token counts, forward under
    the engine's hooks and backward: every call packs its saved tensors and the input
    gradient arrives."""
    torch._dynamo.reset()
    op = CompiledOp("rms_norm_mod", True, skips_frame)
    w = torch.randn(D, dtype=torch.bfloat16, requires_grad=True)
    for n in TOKENS:
        x = torch.randn(n, D, dtype=torch.bfloat16, requires_grad=True)
        shift = torch.randn(2, D, requires_grad=True)
        scale = torch.randn(2, D, requires_grad=True)
        seg = (torch.arange(n) >= n // 2).long()
        before = engine.stats["packed"]
        with engine.block(0):
            y = op(x, w, shift, scale, seg, 1e-6)
        y.float().sum().backward()
        assert engine.stats["packed"] > before and x.grad is not None


def _no_error(fn) -> None:
    try:
        fn()
    except Exception as err:  # noqa: BLE001
        raise AssertionError(f"{type(err).__name__}: {err}") from err


def test_hooks_run_outside_dynamo():
    _no_error(lambda: offloads(ActivationOffload("cpu", min_bytes=0)))
    must_fail(
        lambda: _no_error(lambda: offloads(Traceable("cpu", min_bytes=0))),
        "saved-tensor hooks Dynamo traces",
    )
