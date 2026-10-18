"""DyLoRA on the real backbone (bf16, preset attn-mlp, 4 blocks of rank 4): in eval
the bound forward of an adapted layer is base + the full-rank product at the precision
bar whatever rank a training step drew, and the whole backbone's eval output does not
depend on the draws; in training the drawn rank prefix is used instead."""

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.adapters import AdapterConfig, apply
from kohakumoss.model import load_backbone
from tests._util.error_stats import assert_no_regression
from tests._util.negative import must_fail

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
DTYPE = torch.bfloat16
LAYER = "layers.3.proj.q"
BLOCK = 4
CONFIG = AdapterConfig(
    algo="dylora",
    preset="attn-mlp",
    dim=4 * BLOCK,
    alpha=float(4 * BLOCK),
    algo_kwargs={"block_size": BLOCK},
)


@pytest.fixture(scope="module")
def model(weights_root):
    m, report = load_backbone(weights_root / "base", device="cuda", dtype=DTYPE)
    assert report.ok
    return m.requires_grad_(False)


@pytest.fixture
def adapters(model):
    """DyLoRA on the backbone with every block of every layer perturbed off zero."""
    aset = apply(model, CONFIG, device="cuda")
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for _, p in aset.named_trainable():
            p.add_(torch.randn(p.shape, generator=g).to(p) * 0.02)
    model.bind_ops()
    yield aset
    aset.remove()
    model.train(False)
    model.bind_ops()


def draw(aset, rank: int) -> None:
    """Set every layer's drawn rank, as a training step's draw would."""
    for t in aset.targets:
        t.draws.rank = rank


def full_rank_refs(lin, lyco, x) -> tuple[torch.Tensor, torch.Tensor]:
    """(PyTorch's bf16 output, the fp64 ruler) of base + the full-rank DyLoRA delta."""
    down, up, scale = lyco.get_weight(lyco.lora_dim)
    dw = (up.double() @ down.double()) * float(scale)
    w = lin.base_weight(torch.float32).double() + dw
    b = None if lin.bias is None else lin.bias.double()
    ref = F.linear(x.double(), w, b)
    eager = F.linear(x, w.to(x.dtype), None if b is None else b.to(x.dtype))
    return eager, ref


def test_eval_layer_uses_full_rank(model, adapters):
    lin = model.get_submodule(LAYER)
    lyco = next(t for t in adapters.targets if t.path == LAYER).lyco
    x = torch.randn(64, lin.in_features, device="cuda", dtype=DTYPE)
    eager, ref = full_rank_refs(lin, lyco, x)
    draw(adapters, BLOCK)
    model.train(False)
    with torch.no_grad():
        evaluated = lin(x)
    assert_no_regression(evaluated, eager, ref, name="dylora-eval", test="adapters")
    model.train(True)
    with torch.no_grad():
        trained = lin(x)
    must_fail(
        lambda: assert_no_regression(trained, eager, ref, test="adapters"),
        "a forward over the drawn rank prefix",
    )


def _equal(a, b) -> None:
    assert torch.equal(a, b)


def test_eval_output_ignores_the_draws(model, adapters, real_batch):
    batch = real_batch.on("cuda", DTYPE)

    def outputs() -> list[torch.Tensor]:
        out = []
        for rank in (BLOCK, CONFIG.dim):
            draw(adapters, rank)
            with torch.no_grad():
                out.append(model(*batch).float())
        return out

    model.train(False)
    low, full = outputs()
    _equal(low, full)
    model.train(True)
    low, full = outputs()
    must_fail(lambda: _equal(low, full), "an eval forward that reads the draws")
