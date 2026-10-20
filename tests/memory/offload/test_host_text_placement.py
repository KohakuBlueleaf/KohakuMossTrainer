"""The host text-encoder placement, model-free on constructed linears: decoder layers
whose host form holds another dtype (the bf16 file pages of an fp16 run) keep that form
between passes and are cast per layer during one, so no whole cast copy is held; the
small rest is cast once, the embedding rows at lookup; layers already in the compute
dtype need no ring. A host int8 ConvRot linear is the eager composition as plain ops
(its GEMMs visible to `host_gemm`), over a dense weight built for its layer's turn."""

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from kohakumoss.offload.encoders.text import HostPlacement, TextPlacement
from kohakumoss.offload.weights.layers import module_tensors
from kohakumoss.quant import weights as qw_mod
from kohakumoss.quant.convert import QuantLinear
from kohakumoss.quant.weights import QuantWeight, dequantize, quantize
from kohakumoss_kernels.quant.rotated import quant_linear_rotated_reference
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
DIM = 16
LAYERS = 3
GEMMS = {"linear", "mm", "addmm", "matmul", "bmm"}


def _model() -> nn.Module:
    model = nn.Module()
    model.layers = nn.ModuleList(
        nn.Linear(DIM, DIM, bias=False, dtype=torch.bfloat16) for _ in range(LAYERS)
    )
    model.norm = nn.LayerNorm(DIM, dtype=torch.bfloat16)
    model.embed_tokens = nn.Embedding(10, DIM, dtype=torch.bfloat16)
    return model


def check_host(make) -> None:
    model = _model()
    dtypes = {id(t): torch.float16 for t in module_tensors(model).values()}
    want = [layer.weight.detach().to(torch.float16) for layer in model.layers]
    held = [layer.weight.data_ptr() for layer in model.layers]
    place = make(model, model.layers, model.embed_tokens, "cpu", False, dtypes=dtypes)
    assert [layer.weight.data_ptr() for layer in model.layers] == held
    place.begin()
    for i, layer in enumerate(model.layers):
        place.enter[i]()
        assert layer.weight.dtype == torch.float16
        assert torch.equal(layer.weight, want[i])
        place.leave[i]()
    place.end()
    assert [layer.weight.data_ptr() for layer in model.layers] == held
    assert model.norm.weight.dtype == torch.float16
    rows = place.embed(torch.tensor([1, 2]))
    assert rows.dtype == torch.float16, rows.dtype
    assert model.embed_tokens.weight.dtype == torch.bfloat16


def _cast_all(model, layers, embedding, device, pinned, dtypes):
    return TextPlacement(model, layers, embedding, device, pinned, dtypes=dtypes)


class _Ops(TorchDispatchMode):
    """Records the name of every op dispatched under it."""

    def __init__(self):
        super().__init__()
        self.seen: set[str] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.add(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


def _quant_linear(seed: int) -> QuantLinear:
    g = torch.Generator().manual_seed(seed)
    q, scale = quantize(torch.randn(DIM, DIM, generator=g), "int8")
    base = QuantWeight(q, scale, "int8", rotation=DIM)
    return QuantLinear.from_base(DIM, DIM, base, None, torch.bfloat16)


def check_visible(bind) -> None:
    """A host int8 ConvRot linear equals the eager reference, its GEMMs dispatched
    where `host_gemm` sees them."""
    lin = _quant_linear(0)
    bind(lin.qweight)
    qw, x = lin.qweight, torch.randn(5, DIM, dtype=torch.bfloat16)
    want = quant_linear_rotated_reference(x, qw.q, qw.scale, None, "int8", DIM)
    with _Ops() as ops:
        got = lin(x)
    assert torch.equal(got, want)
    assert ops.seen & GEMMS, ops.seen


def _custom_op(qw: QuantWeight) -> None:
    qw._linear = qw_mod._rotated_eager


def test_host_quant_linear_gemms_are_visible():
    check_visible(lambda qw: qw.bind(torch.bfloat16, "cpu"))
    must_fail(lambda: check_visible(_custom_op), "the opaque custom op on the host")


def check_turns(make) -> None:
    """Each decoder layer's quantized bases are dense for its turn of the pass only."""
    model = _model()
    model.layers = nn.ModuleList(nn.Sequential(_quant_linear(i)) for i in range(LAYERS))
    place = make(model, model.layers, model.embed_tokens, "cpu", False)
    bases = [layer[0].qweight for layer in model.layers]
    place.begin()
    for i, base in enumerate(bases):
        place.enter[i]()
        want = dequantize(base.q, base.scale, torch.bfloat16)
        assert base.dense is not None and torch.equal(base.dense, want)
        assert all(other.dense is None for other in bases if other is not base)
        place.leave[i]()
    place.end()
    assert all(base.dense is None for base in bases)


def _no_turns(model, layers, embedding, device, pinned):
    return TextPlacement(model, layers, embedding, device, pinned)


def test_host_bases_are_dense_for_their_layer_turn():
    check_turns(HostPlacement)
    must_fail(lambda: check_turns(_no_turns), "no dense weight built per layer")


def test_host_layers_are_cast_per_pass():
    check_host(HostPlacement)
    must_fail(lambda: check_host(_cast_all), "every layer held as a cast copy")
    model = _model()
    plain = HostPlacement(model, model.layers, model.embed_tokens, "cpu", False)
    assert plain.streamer is None
