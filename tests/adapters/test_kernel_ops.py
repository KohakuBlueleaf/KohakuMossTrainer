"""Every adapter kernel op on one linear's LyCORIS module (no model): each candidate the
bind can run (the LyCORIS fused kernel of the device's first fused backend, the compiled
body, the eager body) against the eager body under autocast to the test dtype (as
training runs it) and in fp64, forward and backward, at fp16 and bf16."""

from functools import partial

import pytest
import torch
from lycoris.kernels.dispatch import fused_backends
from lycoris.kernels.select import fusable
from lycoris.modules import (
    ButterflyOFTModule,
    DiagOFTModule,
    GLoRAModule,
    IA3Module,
    LoConModule,
    LohaModule,
    LokrModule,
)
from torch import nn

from kohakumoss.adapters.bind import ops, weight_ops
from kohakumoss.adapters.bind.autotuned import body_candidate, compiled, eager
from tests._util.negative import must_fail
from tests._util.oft_ref import fp64_cayley  # noqa: F401  (fixture)
from tests.kernels.blocks._blockops_harness import check

pytestmark = pytest.mark.cuda
OUT, IN, N = 128, 192, 77
DIMS = {"boft": (32, 48)}  # (out, in) of the kinds on a smaller linear
DEV = torch.device("cuda")

MODULES = {
    "lora": lambda lin: LoConModule("l", lin, 1.0, 8, 4.0),
    "dora": lambda lin: LoConModule("l", lin, 1.0, 8, 4.0, weight_decompose=True),
    "loha": lambda lin: LohaModule("l", lin, 1.0, 4, 4.0),
    "lokr": lambda lin: LokrModule("l", lin, 1.0, 4, 4.0),
    "oft": lambda lin: DiagOFTModule("l", lin, 1.0, 16, rescaled=True),
    "boft": lambda lin: ButterflyOFTModule("l", lin, 1.0, 8, rescaled=True),
    "ia3": lambda lin: IA3Module("l", lin, 1.0),
    "ia3_in": lambda lin: IA3Module("l", lin, 1.0, train_on_input=True),
    "glora": lambda lin: GLoRAModule("l", lin, 1.0, 4, 4.0),
}
OPS = {
    "lora.bypass": ("lora", partial(ops.lora_bypass, gamma=0.5)),
    "lora.bypass_with_base": ("lora", partial(ops.lora_bypass_with_base, gamma=0.5)),
    "loha.bypass": ("loha", partial(ops.loha_bypass, gamma=0.5)),
    "loha.bypass_with_base": ("loha", partial(ops.loha_bypass_with_base, gamma=0.5)),
    "lokr.bypass": ("lokr", partial(ops.lokr_bypass, scale=0.5)),
    "lokr.bypass_with_base": ("lokr", partial(ops.lokr_bypass_with_base, scale=0.5)),
    "oft.bypass": ("oft", ops.oft_bypass),
    "oft.rotate": ("oft", ops.oft_rotate),
    "boft.bypass": ("boft", partial(ops.boft_bypass, m=0.7)),
    "boft.rotate": ("boft", partial(ops.boft_rotate, m=0.7)),
    "ia3.scale_out": ("ia3", partial(ops.ia3_bypass_scale, m=0.7)),
    "ia3.scale_in": ("ia3_in", partial(ops.ia3_bypass_scale, m=0.7)),
    "lora.weight": ("lora", weight_ops.lora_weight),
    "lora.merged_weight": ("lora", weight_ops.lora_merged_weight),
    "loha.weight": ("loha", weight_ops.loha_weight),
    "loha.merged_weight": ("loha", weight_ops.loha_merged_weight),
    "lokr.weight": ("lokr", weight_ops.lokr_weight),
    "lokr.merged_weight": ("lokr", weight_ops.lokr_merged_weight),
    "oft.weight": ("oft", weight_ops.oft_weight),
    "oft.merged_weight": ("oft", weight_ops.oft_merged_weight),
    "boft.weight": ("boft", partial(weight_ops.boft_weight, m=0.7)),
    "ia3.weight": ("ia3", partial(weight_ops.ia3_weight, m=0.7)),
    "glora.weight": ("glora", weight_ops.glora_weight),
    "decompose": ("dora", partial(weight_ops.decompose, on_out=True)),
    "decompose_with_delta": (
        "dora",
        partial(weight_ops.decompose_with_delta, on_out=True),
    ),
}
DTYPES = (torch.float16, torch.bfloat16)
pytestmark = [pytestmark, pytest.mark.usefixtures("fp64_cayley")]


def module(kind: str) -> nn.Module:
    """A LyCORIS module of `kind` on a random linear, every tensor random."""
    torch.manual_seed(0)
    out, inp = DIMS.get(kind, (OUT, IN))
    lin = nn.Linear(inp, out).to(DEV).requires_grad_(False)
    lyco = MODULES[kind](lin).to(DEV)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, p in lyco.named_parameters():
            noise = torch.randn(p.shape, generator=g).to(p) * 0.3
            if "oft_blocks" in name:
                p.add_(noise * 0.1)
            elif any(t in name for t in ("dora_scale", "rescale")):
                p.mul_(1 + noise)
            else:
                p.add_(noise)
    return lyco


def problem(name: str, dtype: torch.dtype):
    """(op, operand tuple, which operands take a gradient) of one op at `dtype`."""
    kind, build = OPS[name]
    op = build(module(kind))
    t, _ = next(iter(op.runtime.examples([N], dtype, DEV)))
    operands = op.operands(t)
    grads = [isinstance(a, torch.Tensor) and a.requires_grad for a in operands]
    return op, operands, grads


def _tupled(run, dtype, *ops):
    """`run` on the operand tuple: under autocast to `dtype` as in training, except the
    float64 reference run."""
    if any(isinstance(a, torch.Tensor) and a.dtype == torch.float64 for a in ops):
        return run(ops)
    with torch.autocast(DEV.type, dtype=dtype):
        return run(ops)


def candidates(op, operands: tuple) -> dict:
    """The candidate runs of `op`: the bind's body candidate (compiled, or eager for an
    autograd.Function body), eager, and the fused kernel where the bind would run it
    (the device's first fused backend covers the layer and the operands, see
    `AutotunedOp.resolve`)."""
    impl, body = body_candidate(op, "tokens")
    out = {impl: body, "eager": eager(op)}
    backend = next(iter(fused_backends(DEV)), None)
    if backend is not None and op.in_scope(backend) and fusable(operands):
        out[backend] = partial(op.kernel, backend)
    return out


def check_candidates(name: str, dtype: torch.dtype, runs: dict | None = None) -> None:
    op, operands, grads = problem(name, dtype)
    spec = partial(_tupled, eager(op), dtype)
    for label, run in (runs or candidates(op, operands)).items():
        what = f"{name}/{label}"
        check(partial(_tupled, run, dtype), spec, operands, grads, dtype, what)


@pytest.mark.budget(75)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", sorted(OPS))
def test_every_candidate_matches_the_eager_body(name, dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(
        including_emulation=False
    ):
        pytest.skip("no bf16 on this card")
    check_candidates(name, dtype)


def test_the_kernel_candidate_is_checked():
    """The fused kernel is among the checked candidates wherever one is installed."""
    if not fused_backends(DEV):
        pytest.skip("no fused LyCORIS backend installed")
    op, operands, _ = problem("lora.bypass", torch.float16)
    assert set(candidates(op, operands)) > {"compile", "eager"}


def _scaled(run, ops):
    return run(ops) * 1.1


def _widened(run, ops):
    return run(ops).double()


def test_an_unrelated_output_dtype_is_caught():
    op, _, _ = problem("lora.weight", torch.float16)
    broken = {"compile->fp64": partial(_widened, compiled(op, "tokens"))}
    must_fail(
        lambda: check_candidates("lora.weight", torch.float16, broken), "fp64 output"
    )


@pytest.mark.parametrize("name", ["lora.bypass", "oft.rotate", "lokr.merged_weight"])
def test_a_scaled_candidate_is_caught(name):
    op, _, _ = problem(name, torch.float16)
    broken = {"compile*1.1": partial(_scaled, compiled(op, "tokens"))}
    must_fail(
        lambda: check_candidates(name, torch.float16, broken), f"{name} scaled 10%"
    )
