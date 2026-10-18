"""An autotuned adapter op binds per (token bucket, grad mode): a bucket settled under
no_grad (validation, sampling) does not serve training with the inference winner; a
bound compile candidate is the `CompiledOp` of its operand shapes."""

import pytest
import torch
from lycoris.modules import LoConModule
from torch import nn

from kohakumoss.adapters.bind import autotuned, bodies, ops
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
GAMMA = 0.5


def _layer() -> autotuned.AutotunedOp:
    torch.manual_seed(0)
    lyco = LoConModule("l", nn.Linear(64, 48), 1.0, 4, 4.0)
    return autotuned.AutotunedOp(
        ops.lora_bypass_with_base(lyco, GAMMA), "triton", "tokens"
    )


def _inputs(grad: bool) -> tuple:
    return (
        torch.randn(300, 64, requires_grad=grad),
        torch.randn(300, 48, requires_grad=grad),
    )


@pytest.fixture
def tuning(monkeypatch, tmp_path):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuner, "_TABLES", {})
    monkeypatch.setattr(autotuned, "fusable", lambda operands: True)
    tuner.reset()
    tuner.set_mode("off")
    yield
    tuner.reset()


def _store(op, grad: bool, impl: str) -> None:
    with torch.set_grad_enabled(grad):
        key = autotuned.problem_key(op._operands(_inputs(grad)), op.op.tokens)
    tuner.table(op.name, "cpu").put(key, {"impl": impl}, {}, {})


def check_grad_modes(op) -> None:
    """No-grad stored "compile", grad stored "kernel": each mode runs its own."""
    picks: list[str] = []

    def candidate(name):
        def run(operands):
            picks.append(name)
            return bodies.lora_with_base(operands, GAMMA)

        return run

    op.candidates = {name: candidate(name) for name in op.candidates}
    _store(op, False, "compile")
    _store(op, True, "kernel")
    with torch.no_grad():
        op(_inputs(False))
    op(_inputs(True))
    assert picks == ["compile", "kernel"], picks


def test_bucket_per_grad_mode(tuning):
    check_grad_modes(_layer())


def test_negative_bucket_without_grad_mode(tuning):
    op = _layer()
    op._bucket = lambda t: (autotuned.pow2(t[0].shape[0], 256), t[0].dtype)
    must_fail(lambda: check_grad_modes(op), "inference winner served training")


class _Graph:
    """A `CompiledOp` stand-in running `body` eagerly (its runner is `body`)."""

    def __init__(self, body):
        self.body = body

    def __call__(self, *args):
        return self.body(*args)

    def resolved(self, args):
        return self.body


def check_one_lookup(bind, monkeypatch) -> None:
    """A bound bucket finds its `CompiledOp` once, not per call."""
    made = []
    monkeypatch.setattr(
        autotuned,
        "_graph",
        lambda body, dynamic, shapes: made.append(shapes) or _Graph(body),
    )
    layer = _layer()
    run = bind(autotuned.CompiledBody(layer.op, "tokens"), layer, _inputs(False))
    for n in (40, 300):
        run(layer._operands((torch.randn(n, 64), torch.randn(n, 48))))
    assert len(made) == 1, f"{len(made)} shape lookups for one bound bucket"


def _bound(body, layer, t):
    return body.at(layer._operands(t))


def test_compiled_body_binds_its_shapes(monkeypatch):
    check_one_lookup(_bound, monkeypatch)


def test_negative_lookup_per_call(monkeypatch):
    must_fail(
        lambda: check_one_lookup(lambda body, layer, t: body, monkeypatch),
        "a shape lookup per call",
    )
