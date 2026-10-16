"""Each (kind, mode, multiplier, flags, training) combination binds its specialised callable.

Binding is inspected, not executed, so the fused variants are checked on CPU by binding
with a fused backend name.
"""

from functools import partial

import pytest
import torch
from _adcases import build, make_spec
from test_stored_base import quantized_toy

from kohakumoss.adapters import apply
from kohakumoss.adapters import delta as dl
from kohakumoss.adapters.bind import eager, fused
from kohakumoss.adapters.bind.select import bind_target

pytestmark = pytest.mark.cpu
JOINT = "layers.0.proj.q"
TEXT = "text_embedder.blocks.0.qkv"


def _target(aset, path=JOINT):
    return next(t for t in aset.targets if t.path == path)


def _bind(t, mode="bypass", backend="triton", m=1.0, detach=False, module_p=0.0):
    return bind_target(t, mode, backend, m, detach, module_p)


@pytest.mark.parametrize(
    "name,scalar_fn,diff_fn",
    [
        ("lora", fused.add_base, fused.lora_diff),
        ("lora-scalar", fused.add_base_scalar, fused.lora_diff),
        ("loha", fused.add_base, fused.loha_diff),
        ("lokr", fused.add_base, fused.lokr_diff),
    ],
)
def test_low_rank_bypass_variants(name, scalar_fn, diff_fn):
    _, aset = build(name)
    fn = _bind(_target(aset), m=0.5)
    assert fn.func is scalar_fn
    assert fn.args[-1].func is diff_fn
    gamma = (
        fn.args[-1].args[1] if diff_fn is not fused.lokr_diff else fn.args[-1].args[2]
    )
    assert gamma == pytest.approx(_target(aset).lyco.scale * 0.5)


@pytest.mark.parametrize(
    "name,m,want",
    [
        ("oft", 1.0, fused.oft_bypass),
        ("oft-constraint-rescale", 1.0, fused.oft_bypass_rescaled),
        ("boft", 0.5, fused.boft_bypass),
    ],
)
def test_rotation_bypass_variants(name, m, want):
    _, aset = build(name)
    assert _bind(_target(aset), m=m).func is want


@pytest.mark.parametrize(
    "name,inner",
    [("oft", dl.oft_delta_plain), ("oft-constraint-rescale", dl.oft_delta_rescaled)],
)
def test_oft_at_a_multiplier_binds_the_delta_builder(name, inner):
    _, aset = build(name)
    fn = _bind(_target(aset), m=0.5)
    assert fn.func is eager.delta_bypass
    builder = fn.args[-1]
    assert builder.func is dl._oft_at and builder.args[-1] is inner


@pytest.mark.parametrize(
    "name,backend,want",
    [
        ("ia3", "triton", fused.ia3_bypass_out),
        ("ia3-in", "triton", fused.ia3_bypass_in),
        ("ia3", "torch", eager.ia3_out),
        ("ia3-in", "torch", eager.ia3_in),
    ],
)
def test_ia3_variants(name, backend, want):
    _, aset = build(name)
    assert _bind(_target(aset), backend=backend).func is want


@pytest.mark.parametrize("m,scale", [(1.0, fused.same), (0.5, fused.times)])
@pytest.mark.parametrize(
    "path,lin", [(JOINT, fused.lin_nobias), (TEXT, fused.lin_bias)]
)
def test_merge_variants(m, scale, path, lin):
    _, aset = build("lora-text")
    fn = _bind(_target(aset, path), mode="merge", m=m)
    assert fn.func is fused.merged_forward
    _, _, scale_fn, lin_fn = fn.args
    assert (scale_fn.func if isinstance(scale_fn, partial) else scale_fn) is scale
    assert (lin_fn.func if isinstance(lin_fn, partial) else lin_fn) is lin


@pytest.mark.parametrize("m,mix", [(1.0, fused.keep), (0.5, fused.interpolate)])
def test_dora_fused_variants(m, mix):
    _, aset = build("dora-out")
    fn = _bind(_target(aset), m=m)
    assert fn.func is fused.dora_forward
    mix_fn = fn.args[5]
    assert (mix_fn.func if isinstance(mix_fn, partial) else mix_fn) is mix


@pytest.mark.parametrize(
    "name,detach,want",
    [
        ("dora-out", True, dl.detached_out),
        ("dora-in", True, dl.detached_in),
        ("dora-out", False, dl.dora_merged),
    ],
)
def test_dora_eager_variants(name, detach, want):
    _, aset = build(name)
    fn = _bind(_target(aset), backend="torch", detach=detach)
    assert fn.func is eager.eager_dora
    decompose = fn.args[2]
    got = decompose.args[0] if decompose.func is dl.detached else decompose.func
    assert got is want


@pytest.mark.parametrize("m,builder", [(1.0, dl._unit_only), (0.5, dl._unit_times)])
def test_bind_delta_variants(m, builder):
    _, aset = build("lora")
    assert dl.bind_delta(_target(aset), m).func is builder


def test_bind_delta_dora_and_full():
    _, aset = build("dora-out")
    one, half = dl.bind_delta(_target(aset), 1.0), dl.bind_delta(_target(aset), 0.5)
    assert one.func is dl._dora and one.args[1] is dl._same
    assert half.func is dl._dora and half.args[1].func is dl._times
    _, full = build("lora-text", algo="full", mode="merge")
    assert dl.bind_delta(_target(full, TEXT), 1.0).func is dl._full
    assert dl.bind_delta(_target(full, JOINT), 1.0).func is dl._full_nobias


def test_dylora_rebinds_on_train_and_eval():
    model, aset = build("dylora")
    t = _target(aset)
    model.train()
    assert vars(t.linear)["forward"].func is eager.dylora_train
    model.eval()
    assert vars(t.linear)["forward"].func is eager.delta_bypass


def test_module_dropout_only_binds_while_training():
    model, aset = build("lora", module_dropout=0.3)
    t = _target(aset)
    model.train()
    assert vars(t.linear)["forward"].func is eager.module_dropout
    model.eval()
    assert vars(t.linear)["forward"].func is not eager.module_dropout


def test_stored_base_binds_bypass_for_merge_mode():
    model = quantized_toy()
    torch.manual_seed(5)
    aset = apply(model, make_spec("lora", mode="merge"))
    fn = _bind(_target(aset), mode="merge")
    assert fn.func is fused.add_base
