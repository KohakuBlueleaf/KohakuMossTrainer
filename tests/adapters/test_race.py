"""Layers bound to LyCORIS's tier race: one `TierSlot` per op, resolved once per bucket.

The CPU tests check the binding (auto backend -> slots, pinned backend -> direct fused
calls), the slot's resolve-once caching, and the mirrored ops' eager arms against the
eager applier (CPU operands make LyCORIS pick its torch body). The CUDA test checks a
real race settles and is not asked again.
"""

from functools import partial

import pytest
import torch
from _adcases import build
from lycoris.kernels import tier

from kohakumoss.adapters.bind import fused, race
from kohakumoss.adapters.bind.race import Op, TierSlot
from kohakumoss.adapters.bind.select import bind_target
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
JOINT = "layers.0.proj.q"
BYPASS_OPS = {
    "lora": "locon.bypass",
    "loha": "loha.bypass",
    "lokr": "lokr.bypass",
    "oft": "diag_oft.bypass",
    "oft-constraint-rescale": "diag_oft.bypass",
    "boft": "boft.bypass",
}
MERGE_OPS = {
    "lora": "locon.merge",
    "loha": "loha.merge",
    "lokr": "lokr.merge",
    "oft": "diag_oft.merge",
    "boft": "boft.merge",
}
NUMERIC = (
    "lora",
    "lora-scalar",
    "loha",
    "lokr",
    "oft",
    "oft-constraint-rescale",
    "boft",
    "dora-out",
    "doha",
)


def _target(aset):
    return next(t for t in aset.targets if t.path == JOINT)


def _bind(t, mode="bypass", backend="triton", m=1.0, race_on=True):
    return bind_target(t, mode, backend, m, False, 0.0, race=race_on)


def _slot_of(fn) -> TierSlot:
    """The TierSlot inside a bound forward (bypass: add_base / post_add over flat)."""
    inner = fn.args[-1]
    return inner.args[0] if inner.func is race.flat else inner


@pytest.mark.parametrize("name,op", BYPASS_OPS.items())
def test_auto_backend_binds_a_slot_per_bypass_op(name, op):
    _, aset = build(name)
    fn = _bind(_target(aset), m=0.5 if name == "boft" else 1.0)
    want = fused.post_add if "oft" in op else fused.add_base
    assert fn.func is want
    slot = _slot_of(fn)
    assert isinstance(slot, TierSlot) and slot.op.name == op
    assert slot.bucket is race.token_bucket


@pytest.mark.parametrize("name,op", MERGE_OPS.items())
def test_auto_backend_binds_a_slot_per_merge_op(name, op):
    _, aset = build(name)
    fn = _bind(_target(aset), mode="merge")
    assert fn.func is fused.merged_forward
    slot = fn.args[1]
    assert isinstance(slot, TierSlot) and slot.op.name == op
    assert slot.bucket is race.static_bucket


def test_dora_binds_a_decompose_slot():
    _, aset = build("dora-out")
    fn = _bind(_target(aset))
    assert fn.func is fused.dora_forward
    assert fn.args[2].op.name == "locon.merge"
    assert fn.args[3].op.name == "weight_decompose"


def test_pinned_backend_binds_the_fused_kernel_directly():
    _, aset = build("lora")
    fn = _bind(_target(aset), race_on=False)
    assert fn.args[-1].func is fused.lora_diff
    assert fn.args[-1].args[2] == "triton"


def test_loha_rank_product_is_raced():
    _, aset = build("loha")
    op = _slot_of(_bind(_target(aset))).op
    assert set(op.extras) == {"rank_product"}


@pytest.mark.parametrize("m", [1.0, 0.5])
@pytest.mark.parametrize("mode", ["bypass", "merge"])
@pytest.mark.parametrize("name", NUMERIC)
def test_raced_arms_match_the_eager_applier(name, mode, m):
    if name.startswith("oft") and m != 1.0:
        pytest.skip("OFT at a multiplier binds the eager delta builder")
    _, aset = build(name)
    t = _target(aset)
    x = torch.randn(2, 7, t.lyco.shape[1])
    with torch.no_grad():
        want = _bind(t, mode, "torch", m, race_on=False)(x)
        got = _bind(t, mode, "triton", m)(x)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def _counting(monkeypatch):
    calls = []
    real = tier.best

    def best(*args):
        calls.append(args[1][0].shape)
        return real(*args)

    monkeypatch.setattr(tier, "best", best)
    return calls


def _toy_op() -> Op:
    w = torch.randn(4, 3)
    return Op(
        "test.bypass",
        partial(_pair, w),
        True,
        _never,
        torch.nn.functional.linear,
        _same,
    )


def _pair(w, x):
    return (x, w)


def _same(ops):
    return ops


def _never(backend, ops):
    raise AssertionError("no fused arm on CPU operands")


def _tensor_ids(args) -> set:
    return {id(a) for a in args if isinstance(a, torch.Tensor)}


@pytest.mark.parametrize("mode", ["bypass", "merge"])
@pytest.mark.parametrize("name", NUMERIC)
def test_race_arguments_share_one_operand_tuple(name, mode):
    """The fp64 gate maps gradients by identity: every operand LyCORIS keys on must
    be the very object the eager body receives (fp64 input: every factor is cast)."""
    _, aset = build(name)
    t = _target(aset)
    slots = _slots(_bind(t, mode, m=1.0))
    assert slots
    out_dim, in_dim = t.lyco.shape[:2]
    for slot in slots:
        name = slot.op.name
        if not name.endswith(".bypass"):
            probe = t.base_fn(torch.float64)
        else:
            dim = out_dim if "oft" in name else in_dim
            probe = torch.randn(9, dim, dtype=torch.float64)
        args = race.race_args(slot.op, probe)
        assert _tensor_ids(args[1]) <= _tensor_ids(args[6]), slot.op.name


def test_recast_operands_are_caught():
    _, aset = build("lora")
    op = _slot_of(_bind(_target(aset))).op
    x = torch.randn(9, _target(aset).lyco.shape[1], dtype=torch.float64)

    def recast():
        broken = Op(
            op.name,
            op.operands,
            op.supported,
            op.fused,
            op.ref,
            lambda ops: (*op.operands(ops[0]), None, 1.0, {}),
        )
        args = race.race_args(broken, x)
        assert _tensor_ids(args[1]) <= _tensor_ids(args[6])

    must_fail(recast, "eager-body arguments cast separately from the operands")


def _slots(fn) -> list:
    if isinstance(fn, TierSlot):
        return [fn]
    if not isinstance(fn, partial):
        return []
    found = _slots(fn.func)
    for a in fn.args:
        found += _slots(a)
    return found


def test_slot_resolves_once_per_bucket(monkeypatch):
    calls = _counting(monkeypatch)
    slot = TierSlot(_toy_op(), race.token_bucket)
    for rows in (100, 120, 100, 300, 260):
        slot(torch.randn(rows, 3))
    assert len(calls) == 2, calls
    assert len(slot.arms) == 2
    with torch.no_grad():
        slot(torch.randn(100, 3))
    assert len(calls) == 3


def test_a_slot_ignoring_the_bucket_is_caught(monkeypatch):
    calls = _counting(monkeypatch)
    slot = TierSlot(_toy_op(), lambda t: ())

    def one_key_for_all():
        for rows in (100, 300):
            slot(torch.randn(rows, 3))
        assert len(calls) == 2

    must_fail(one_key_for_all, "a slot keyed without the token bucket")


def test_unsettled_picks_are_asked_every_call(monkeypatch):
    calls = _counting(monkeypatch)
    monkeypatch.setattr(race, "settled", lambda *a: False)
    slot = TierSlot(_toy_op(), race.token_bucket)
    for _ in range(3):
        slot(torch.randn(16, 3))
    assert len(calls) == 3 and not slot.arms
    monkeypatch.setattr(race, "settled", lambda *a: True)
    slot(torch.randn(16, 3))
    slot(torch.randn(16, 3))
    assert len(calls) == 4 and len(slot.arms) == 1


def test_arms_by_pick():
    op = _toy_op()
    x = torch.randn(5, 3)
    torch.testing.assert_close(race.arm(op, "torch")(x), x @ op.operands(x)[1].T)
    assert race.body(op, "compile").func is race._compiled_body
    _, aset = build("loha")
    loha = _slot_of(_bind(_target(aset))).op
    assert race.body(loha, "rank_product") is loha.extras["rank_product"]
    assert race.body(loha, "tilelang").args[-1] == "tilelang"


@pytest.mark.cuda
def test_a_real_race_settles_and_is_not_asked_again(monkeypatch):
    _, aset = build("lora")
    t = _target(aset)
    t.lyco.cuda()
    slot = _slot_of(_bind(t, backend="triton"))
    x = torch.randn(256, t.lyco.shape[1], device="cuda", dtype=torch.float16)
    x.requires_grad_(True)
    slot(x)
    key = slot.bucket(x)
    assert key in slot.arms
    calls = _counting(monkeypatch)
    slot(x).float().sum().backward()
    assert not calls
