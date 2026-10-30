"""STREAM_LOSS_PROBE: parameter groups of MOSS names, per-term gradient norms (total and per
group) and cosines against direct autograd, the fp16 overflow step-down, and the probe
schedule."""

import math

import pytest
import torch
from torch import nn

from kohakumoss.training.grad_probe import GradProbe, build_probe, group_of
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]


def test_group_of():
    assert group_of("video_embedder.weight") == "embed"
    assert group_of("text_embedder.blocks.0.ffn_down.weight") == "embed"
    assert group_of("layers.7.proj.o.weight") == "attn_out"
    assert group_of("layers.7.proj.ffn_down.weight") == "ffn_out"
    assert group_of("layers.7.proj.qkv.weight") == "in_proj"
    assert group_of("layers.7.proj.ffn_up_gate.weight") == "in_proj"
    assert group_of("layers.7.norm1.weight") == "norm"
    assert group_of("shared_block_modulation.weight") == "cond"
    assert group_of("final_linear_video.weight") == "head"
    assert group_of("rope.bands") == "other"


def _named(dtype=torch.float32):
    torch.manual_seed(0)
    return [
        ("video_embedder.weight", nn.Parameter(torch.randn(4, 3, dtype=dtype))),
        ("layers.0.proj.o.weight", nn.Parameter(torch.randn(5, dtype=dtype))),
        (
            "frozen.weight",
            nn.Parameter(torch.randn(2, dtype=dtype), requires_grad=False),
        ),
    ]


def _check(measure) -> None:
    named = _named()
    a, b = named[0][1], named[1][1]
    x = torch.linspace(-1, 1, 3)
    task = (a @ x).sum() + b.sum()
    term = a.square().sum() * 0.5 + (b**3).sum()
    got = measure(named, {"task": task, "rms": term})
    ga, gb = torch.autograd.grad(task, (a, b), retain_graph=True)
    ta, tb = torch.autograd.grad(term, (a, b))
    gt = torch.cat([ga.flatten(), gb.flatten()])
    tt = torch.cat([ta.flatten(), tb.flatten()])
    assert got["grad_task"].item() == pytest.approx(gt.norm().item(), rel=1e-5)
    assert got["grad_rms"].item() == pytest.approx(tt.norm().item(), rel=1e-5)
    assert got["grad_rms_embed"].item() == pytest.approx(ta.norm().item(), rel=1e-5)
    assert got["grad_rms_attn_out"].item() == pytest.approx(tb.norm().item(), rel=1e-5)
    cos = torch.dot(gt, tt) / (gt.norm() * tt.norm())
    assert got["grad_cos_rms"].item() == pytest.approx(cos.item(), rel=1e-5)


def test_measure():
    _check(lambda named, values: GradProbe(named, 1).measure(values))


def test_measure_negative():
    def unscaled(named, values):
        probe = GradProbe(named, 1)
        out = probe.measure(values)
        return {
            k: v * 2.0**16 if k.startswith("grad_") and "cos" not in k else v
            for k, v in out.items()
        }

    must_fail(lambda: _check(unscaled), "norms left at the probe scale")


def test_fp16_step_down():
    """A term whose fp16 gradient overflows at the first scale is measured at a lower one."""
    p = nn.Parameter(torch.full((4,), 2.0, dtype=torch.float16))
    probe = GradProbe([("layers.0.proj.o.weight", p)], 1)
    task = p.float().sum()
    big = (p.float() * 1000.0).sum()
    out = probe.measure({"task": task, "peak": big})
    assert out["grad_task"].item() == pytest.approx(2.0, rel=1e-3)
    assert out["grad_peak"].item() == pytest.approx(2000.0, rel=1e-3)
    assert math.isfinite(out["grad_cos_peak"].item())


def test_schedule_and_build():
    donated = torch._functorch.config.donated_buffer
    assert build_probe({"STREAM_LOSS_PROBE": 0}, nn.Linear(2, 2)) is None
    assert torch._functorch.config.donated_buffer == donated
    try:
        build_probe({"STREAM_LOSS_PROBE": 3}, nn.Linear(2, 2))
        assert torch._functorch.config.donated_buffer is False
    finally:
        torch._functorch.config.donated_buffer = donated
    probe = build_probe({"STREAM_LOSS_PROBE": 5}, nn.Linear(2, 2))
    assert [s for s in range(12) if probe.due(s)] == [5, 10]
    every = build_probe({"STREAM_LOSS_PROBE": 1}, nn.Linear(2, 2))
    assert [s for s in range(4) if every.due(s)] == [1, 2, 3]
    assert len(probe.params) == 2
