"""Adapters on a quantized base, one linear (no model): the bound training forward and
the inference attach both run bypass on the base's own forward, never read a dense
base weight (DoRA reads one transient copy for its norm), and match an independent fp64
formula of the adapter through the error statistics, with the same adapter on a dense base as
the baseline. ADAPTER_MODE='merge' on such a base runs bypass with a WARNING, in
training and in inference; a dense-diff file is refused."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kohakumoss.adapters import AdapterConfig, AdapterStack, apply
from kohakumoss.model.linear import MossLinear
from kohakumoss.quant.weights import QuantWeight, dequantize
from tests._util.error_stats import assert_no_regression
from tests._util.negative import must_fail
from tests._util.oft_ref import butterfly, cayley

pytestmark = pytest.mark.cpu
OUT, IN, N = 64, 96, 13
PATH = "layers.0.proj.q"
KINDS = {
    "lora": dict(algo="lora", dim=8, alpha=4.0),
    "loha": dict(algo="loha", dim=4, alpha=4.0),
    "lokr": dict(
        algo="lokr",
        dim=10000,
        alpha=1.0,
        algo_kwargs={"factor": 16, "full_matrix": True},
    ),
    "oft": dict(algo="oft", dim=16),
    "boft": dict(algo="boft", dim=8),
    "ia3": dict(algo="ia3", dim=1),
    "glora": dict(algo="glora", dim=4, alpha=4.0),
    "dora": dict(algo="lora", dim=8, alpha=8.0, dora=True),
    "tlora": dict(algo="tlora", dim=8, alpha=8.0),
}
BASES = ("int8", "fp8")


def holder(lin: nn.Module) -> nn.Module:
    """The linear at a MOSS module path (a container, no forward)."""
    block = nn.Module()
    block.proj = nn.ModuleDict({"q": lin})
    root = nn.Module()
    root.layers = nn.ModuleList([block])
    return root


def dense_linear(w, b) -> MossLinear:
    lin = MossLinear(IN, OUT, bias=True)
    with torch.no_grad():
        lin.weight.copy_(w)
        lin.bias.copy_(b)
    return lin


def stored_linear(w, b, fmt: str) -> MossLinear:
    """The layout quant.convert leaves: an empty weight placeholder plus the storage."""
    lin = dense_linear(w, b)
    lin.weight = nn.Parameter(torch.empty(0), requires_grad=False)
    lin.set_base(QuantWeight.from_weight(w, fmt))
    return lin


def problem(fmt: str, seed: int = 0):
    """(weight, bias, the stored linear, the dequantized weight it holds, input)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(OUT, IN, generator=g) * 0.1
    b = torch.randn(OUT, generator=g) * 0.1
    lin = stored_linear(w, b, fmt)
    deq = dequantize(lin.base.q, lin.base.scale, torch.float32)
    x = torch.randn(N, IN, generator=g)
    return deq, b, lin, x


def adapted(lin: nn.Module, kind: str, mode: str = "bypass"):
    root = holder(lin)
    root.requires_grad_(False)
    config = AdapterConfig(preset="attn", mode=mode, backend="torch", **KINDS[kind])
    return apply(root, config)


def perturb(aset, seed: int = 3) -> None:
    """Every trainable tensor random (LyCORIS zero-inits one factor)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in aset.named_trainable():
            noise = torch.randn(p.shape, generator=g).to(p) * 0.3
            if any(t in name for t in ("dora_scale", "rescale", "lambda_layer")):
                p.mul_(1 + noise)
            elif "oft_blocks" in name:
                p.add_(noise * 0.1)
            else:
                p.add_(noise)


def _d(t) -> torch.Tensor:
    return t.detach().double()


def reference(kind: str, lyco, w, b, x) -> torch.Tensor:
    """The fp64 output of base (w, b) plus the adapter, from its tensors (LyCORIS's
    bypass semantics: the rotation / channel kinds act on the biased base output)."""
    w, b, x = w.double(), b.double(), x.double()
    y = x @ w.T + b
    s = float(lyco.scale) if hasattr(lyco, "scale") else 1.0
    match kind:
        case "lora":
            dw = _d(lyco.lora_up.weight) @ _d(lyco.lora_down.weight) * s
        case "loha":
            h1 = _d(lyco.hada_w1_a) @ _d(lyco.hada_w1_b)
            h2 = _d(lyco.hada_w2_a) @ _d(lyco.hada_w2_b)
            dw = h1 * h2 * s
        case "lokr":
            dw = torch.kron(_d(lyco.lokr_w1), _d(lyco.lokr_w2)) * s
        case "glora":
            a = _d(lyco.a1.weight) @ _d(lyco.a2.weight)
            dw = (w @ a + _d(lyco.b1.weight) @ _d(lyco.b2.weight)) * s
        case "dora":
            dw = _d(lyco.lora_up.weight) @ _d(lyco.lora_down.weight) * s
            merged = w + dw
            mag = _d(lyco.dora_scale).reshape(-1, 1)
            return x @ (merged * mag / merged.norm(dim=1, keepdim=True)).T + b
        case "tlora":
            q, p, lam = (
                _d(lyco.q_layer.weight),
                _d(lyco.p_layer.weight),
                _d(lyco.lambda_layer),
            )
            q0, p0, lam0 = _d(lyco.base_q), _d(lyco.base_p), _d(lyco.base_lambda)
            dy = ((x @ q.T) * lam) @ p.T - ((x @ q0.T) * lam0) @ p0.T
            return y + dy * s
        case "ia3":
            return y * (1 + _d(lyco.weight))
        case "oft":
            blocks = _d(lyco.oft_blocks)
            k, n, _ = blocks.shape
            r = cayley(blocks) - torch.eye(n, dtype=torch.float64)
            dy = torch.einsum("k n m, t k n -> t k m", r, y.reshape(-1, k, n))
            return y + dy.reshape(y.shape)
        case "boft":
            return butterfly(cayley(_d(lyco.oft_blocks)), y)
    return y + x @ dw.T


def eager_baseline(kind, aset, w, b, x) -> torch.Tensor:
    """The same adapter on a dense fp32 base holding the dequantized weight."""
    dense = dense_linear(w, b)
    twin = adapted(dense, kind)
    twin.load_state_dict(aset.state_dict())
    return dense(x)


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_training_bind_is_bypass_on_a_stored_base(kind, base):
    w, b, lin, x = problem(base)
    aset = adapted(lin, kind)
    perturb(aset)
    lyco = aset.targets[0].lyco
    with torch.no_grad():
        out = lin(x)
        ref = reference(kind, lyco, w, b, x)
        eager = eager_baseline(kind, aset, w, b, x)
        assert_no_regression(out, eager, ref, name=f"{kind}-{base}", test="adapters")
        plain = F.linear(x, w, b)
    must_fail(
        lambda: assert_no_regression(
            plain, eager, ref, name="no adapter", test="adapters"
        ),
        "the adapter's delta dropped",
    )


def _counting(lin) -> list:
    """Count reads of the stored weight outside the base's own forward."""
    calls = []
    base = lin.base
    inner = base.weight

    def weight(dtype):
        calls.append(dtype)
        return inner(dtype)

    def linear(x, bias):
        return F.linear(x, inner(x.dtype), bias)

    base.weight = weight
    base.linear = linear
    lin.rebind()
    return calls


def _forward_reads(kind: str) -> list:
    """The stored-weight reads of one bound forward (construction reads excluded)."""
    w, b, lin, x = problem("int8")
    calls = _counting(lin)
    aset = adapted(lin, kind)
    perturb(aset)
    calls.clear()
    with torch.no_grad():
        lin(x)
    return calls


@pytest.mark.parametrize(
    "kind", ["lora", "loha", "lokr", "oft", "boft", "glora", "tlora"]
)
def test_bypass_never_reads_the_dense_base(kind):
    calls = _forward_reads(kind)
    assert calls == [], f"{kind} read the dense base {len(calls)} times"


def test_the_read_counter_sees_dora():
    calls = _forward_reads("dora")
    assert len(calls) == 1, calls
    must_fail(lambda: _no_reads(calls), "DoRA's transient norm read")


def _no_reads(calls) -> None:
    assert calls == []


def _warned_bypass(caplog, build) -> None:
    caplog.clear()
    with caplog.at_level("WARNING", logger="kohakumoss"):
        build()
    assert any("run as bypass" in r.getMessage() for r in caplog.records), caplog.text


def test_merge_on_a_stored_base_runs_bypass(caplog):
    w, b, lin, x = problem("int8")
    _warned_bypass(caplog, lambda: adapted(lin, "lora", mode="merge"))
    merged = adapted(problem("int8")[2], "lora", mode="merge")
    assert merged.targets[0].stored
    must_fail(
        lambda: _warned_bypass(
            caplog, lambda: adapted(dense_linear(w, b), "lora", "merge")
        ),
        "a dense base merges",
    )


@pytest.mark.parametrize("mode", ["bypass", "merge"])
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_inference_attach_on_a_stored_base(kind, mode):
    w, b, lin, x = problem("int8")
    aset = adapted(lin, kind)
    perturb(aset)
    lyco = aset.targets[0].lyco
    with torch.no_grad():
        ref = reference(kind, lyco, w, b, x)
        eager = eager_baseline(kind, aset, w, b, x)
    tensors = aset.tensors("lycoris")
    _, _, fresh, _ = problem("int8")
    root = holder(fresh)
    stack = AdapterStack(root, mode=mode)
    stack.add(tensors, 1.0)
    with torch.no_grad():
        out = fresh(x)
    assert_no_regression(out, eager, ref, name=f"infer-{kind}-{mode}", test="adapters")
    assert fresh.base is not None and fresh.weight.numel() == 0
    stack.remove()
    with torch.no_grad():
        assert torch.equal(fresh(x), F.linear(x, w, b))
    must_fail(
        lambda: assert_no_regression(F.linear(x, w, b), eager, ref, test="adapters"),
        "the loaded adapter's delta dropped",
    )


def test_a_dense_diff_file_is_refused_on_a_stored_base():
    w, b, lin, x = problem("int8")
    stack = AdapterStack(holder(lin), mode="bypass")
    diff = {f"lora_unet_{PATH.replace('.', '_')}.diff": torch.zeros(OUT, IN)}
    with pytest.raises(ValueError, match="dense-diff adapter needs a 16/32-bit"):
        stack.add(diff, 1.0)
    dense = AdapterStack(holder(dense_linear(w, b)), mode="bypass")
    dense.add(diff, 1.0)
