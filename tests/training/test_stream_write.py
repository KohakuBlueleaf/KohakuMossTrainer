"""STREAM_LOSS write term: the annealed hinge share, the per-sublayer write energy (the
residual scale undone), the parts filter, the collection window, the shrink-only
gradient of `dilute=False`, and a checkpoint recompute matching its forward."""

import pytest
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from kohakumoss.training.stream_loss import (
    StreamPenalty,
    spec_of,
    top_shares,
    write_energy,
    write_target,
)
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]


class Tapped(nn.Module):
    """A block stand-in: adds `wa` after scaling its input by `r`, then adds `wf`,
    calling the write taps as JointBlock does."""

    def __init__(self, wa: Tensor, wf: Tensor, r: float = 1.0):
        super().__init__()
        self.wa, self.wf, self.r = wa, wf, r
        self.ta = self.tf = lambda *a: None

    def bind_write_taps(self, attn, ffn) -> None:
        self.ta, self.tf = attn, ffn

    def forward(self, x: Tensor) -> Tensor:
        a = x * self.r + self.wa
        self.ta(a, x, self.r)
        out = a + self.wf
        self.tf(out, a, 1.0)
        return out


def _write(spike: float, rows: int = 4, dim: int = 8) -> Tensor:
    """Rows of ones with channel 0 at `spike`: the top-1 share is s^2 / (s^2 + dim - 1)."""
    w = torch.ones(rows, dim)
    w[:, 0] = spike
    return w


def _share(spike: float, dim: int = 8) -> float:
    return spike**2 / (spike**2 + dim - 1)


def _spec(**write) -> dict:
    return spec_of({"STREAM_LOSS": {"write": {"weight": 2.0, "k": 1, **write}}})


def test_write_target_schedule():
    term = {"target": 0.4, "anneal": 100, "start": 50}
    assert write_target(term, 0) == 1.0
    assert write_target(term, 49) == 1.0
    assert write_target(term, 50) == pytest.approx(1.0)
    assert write_target(term, 100) == pytest.approx(0.7)
    assert write_target(term, 150) == pytest.approx(0.4)
    assert write_target(term, 10_000) == pytest.approx(0.4)
    assert write_target({"target": 0.4, "anneal": 0, "start": 0}, 0) == 0.4


def test_write_energy_undoes_the_residual_scale():
    x = torch.randn(4, 8)
    w = _write(3.0)
    e = write_energy(x * 0.25 + w, x, 0.25)
    assert torch.allclose(e, w.square().sum(0), atol=1e-5)


def _value(layers, spec, step=0):
    pen = StreamPenalty(nn.ModuleList(layers), spec, (1.0,) * len(layers))
    x = torch.randn(4, 8)
    with pen.collecting():
        h = x
        for layer in layers:
            h = layer(h)
    return pen.take(None, step)


def check_write_value(layers_of) -> None:
    """Two blocks, attention writes with top-1 shares a0, a1 and FFN writes f0, f1, a
    residual scale 0.5: the hinge mean over the four sublayers at target 0.5, the part
    means and the target as metrics, weight 2 in the total."""
    spikes = {"a": (3.0, 1.0), "f": (2.0, 4.0)}
    layers = layers_of(spikes)
    total, parts, terms = _value(layers, _spec(target=0.5))
    shares = [_share(s) for s in spikes["a"] + spikes["f"]]
    want = sum(max(s - 0.5, 0.0) ** 2 for s in shares) / 4
    assert terms["write"].item() == pytest.approx(want, rel=1e-5)
    assert total.item() == pytest.approx(2.0 * want, rel=1e-5)
    assert parts["write_share_attn"].item() == pytest.approx(
        sum(shares[:2]) / 2, rel=1e-5
    )
    assert parts["write_share_ffn"].item() == pytest.approx(
        sum(shares[2:]) / 2, rel=1e-5
    )
    assert parts["write_target"].item() == pytest.approx(0.5)


def _layers(spikes, r=0.5):
    return [Tapped(_write(a), _write(f), r) for a, f in zip(spikes["a"], spikes["f"])]


def test_write_value():
    check_write_value(_layers)


def test_write_value_negative():
    class Unscaled(Tapped):
        def forward(self, x):
            a = x * self.r + self.wa
            self.ta(a, x, 1.0)
            out = a + self.wf
            self.tf(out, a, 1.0)
            return out

    must_fail(
        lambda: check_write_value(
            lambda s: [
                Unscaled(_write(a), _write(f), 0.5) for a, f in zip(s["a"], s["f"])
            ]
        ),
        "write taken as out - x with a residual scale",
    )


def test_write_parts_and_anneal():
    spikes = {"a": (3.0, 1.0), "f": (2.0, 4.0)}
    _, parts, terms = _value(_layers(spikes), _spec(target=0.5, parts="ffn"))
    assert "write_share_attn" not in parts
    want = sum(max(_share(s) - 0.5, 0.0) ** 2 for s in spikes["f"]) / 2
    assert terms["write"].item() == pytest.approx(want, rel=1e-5)
    _, parts, terms = _value(_layers(spikes), _spec(target=0.5, anneal=10, start=5), 3)
    assert terms["write"].item() == 0.0 and parts["write_target"].item() == 1.0


def test_write_collects_only_inside_the_window():
    layers = _layers({"a": (3.0,), "f": (2.0,)})
    pen = StreamPenalty(nn.ModuleList(layers), _spec(target=0.5), (1.0,))
    x = torch.randn(4, 8)
    layers[0](x)
    assert pen.writes == {"attn": [], "ffn": []}
    with torch.no_grad(), pen.collecting():
        layers[0](x)
    assert pen.writes == {"attn": [], "ffn": []}
    with pytest.raises(RuntimeError):
        pen.take(None, 0)


def check_shrink_only(dilute: bool) -> None:
    """With the total detached, the gradient of the share is zero on every channel
    outside the top k and shrinks the top channel."""
    w = _write(3.0).requires_grad_(True)
    e = w.square().sum(0)
    (g,) = torch.autograd.grad(top_shares(e[None], 1, dilute)[0], w)
    assert torch.count_nonzero(g[:, 1:]) == 0
    assert (g[:, 0] * w[:, 0] > 0).all()


def test_write_shrink_only():
    check_shrink_only(False)


def test_write_shrink_only_negative():
    must_fail(
        lambda: check_shrink_only(True), "the full total grows the other channels"
    )


def _grads(tap_mode: str, ckpt: bool) -> list[Tensor]:
    """Gradients of the write term over two blocks' write tensors, the blocks run
    through non-reentrant checkpoints or plainly."""
    wa = [_write(s).requires_grad_(True) for s in (3.0, 2.0)]
    wf = [_write(s).requires_grad_(True) for s in (1.5, 4.0)]
    layers = [Tapped(a, f) for a, f in zip(wa, wf)]
    pen = StreamPenalty(nn.ModuleList(layers), _spec(target=0.1), (1.0, 1.0))
    if tap_mode == "forward_only":

        def tap(part, out, x, r):
            if pen.on:
                pen.writes[part].append(write_energy(out, x, r))

        for layer in layers:
            layer.bind_write_taps(lambda *a: tap("attn", *a), lambda *a: tap("ffn", *a))
    h = torch.randn(4, 8, requires_grad=True)
    with pen.collecting():
        for layer in layers:
            h = checkpoint(layer, h, use_reentrant=False) if ckpt else layer(h)
    _, _, terms = pen.take(None, 0)
    return list(torch.autograd.grad(terms["write"], wa + wf))


def check_recompute(tap_mode: str) -> None:
    try:
        got = _grads(tap_mode, True)
    except torch.utils.checkpoint.CheckpointError as err:
        raise AssertionError(f"checkpoint recompute failed: {err}") from err
    for g, want in zip(got, _grads(tap_mode, False)):
        assert torch.allclose(g, want, atol=1e-6)


def test_write_checkpoint_recompute():
    check_recompute("always")


def test_write_checkpoint_recompute_negative():
    must_fail(
        lambda: check_recompute("forward_only"), "energy computed only while collecting"
    )
