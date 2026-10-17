"""Every memory-knob combination trains the tiny model like the plain reference.

Axes: checkpoint x activation offload x stream x gradient offload x GRAD_ACC x
optimiser x base precision. `stream` decides what trains: `swap` and `off` with a
compute-dtype base train everything (full FT); `ring` and quantized bases train the
non-block parameters through frozen blocks. Invalid combinations must be rejected by
`resolve` with a message.
"""

import itertools

import pytest
import torch

from kohakumoss.offload.gpu_adamw import MasterAdamW
from kohakumoss.training.memory.profiles import resolve
from kohakumoss.training.memory.session import MemorySession
from kohakumoss.training.memory.synthetic import SyntheticModel, synthetic_batch
from kohakumoss_kernels.hardware.arch import arch_of
from tests._util.negative import must_fail

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCH = arch_of(DEVICE)
CHECKPOINTS = ("none", "block", "sub-block", "selective", "partial:0.5")
STREAMS = ("off", "ring", "swap")
OPTIMS = ("gpu", "stream", "cpu")
BASES = ["fp32", "int8"]
if DEVICE == "cuda":
    BASES = ["fp16", "int8"] + (["bf16"] if ARCH.has_bf16 else [])
    BASES += ["fp8"] if ARCH.has_fp8 else []
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
TOL = {torch.float32: 1e-4, torch.float16: 3e-2, torch.bfloat16: 5e-2}
UPDATE_TOL = {torch.float32: 1e-4, torch.float16: 5e-2, torch.bfloat16: 5e-2}
LR = 1e-2
STEPS = 2


def _combos():
    for ck, act, st, go, acc, opt, base in itertools.product(
        CHECKPOINTS, (False, True), STREAMS, (False, True), (1, 4), OPTIMS, BASES
    ):
        yield ck, act, st, go, acc, opt, base


def _setup(base, stream):
    quant = base in ("int8", "fp8")
    dtype = (
        (torch.float16 if DEVICE == "cuda" else torch.float32)
        if quant
        else DTYPES[base]
    )
    full = stream == "swap" or (stream == "off" and not quant)
    return dtype, full, quant


def _model(dtype, full):
    torch.manual_seed(0)
    model = SyntheticModel(32, 64, 4, 8).to(DEVICE, dtype)
    if not full:
        for p in model.layers.parameters():
            p.requires_grad_(False)
    return model


def _batches(dtype, acc):
    return [
        synthetic_batch([(3, 4), (2, 3)], [4, 2], 8, DEVICE, dtype, seed=10 + i)
        for i in range(acc)
    ]


def _loss(model, data):
    img, txt, sigma, seq, target = data
    return (model(img, txt, sigma, seq).float() - target.float()).square().mean()


def _reference(dtype, full, quant, base, acc):
    model = _model(dtype, full)
    if quant:
        plan = resolve(
            "balanced", {"checkpoint": "none", "base_precision": base}, DEVICE
        )
        MemorySession(model, plan, DEVICE, dtype, quant_min_features=0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = MasterAdamW(params, lr=LR)
    losses = []
    for _ in range(STEPS):
        for data in _batches(dtype, acc):
            loss = _loss(model, data)
            loss.backward()
            losses.append(loss.item())
        opt.step()
        opt.zero_grad()
    names = {id(p): n for n, p in model.named_parameters()}
    masters = {names[pid]: s[0].float().cpu() for pid, s in opt.state.items()}
    return losses, masters, model


def _overrides(ck, act, st, go, opt, base):
    quant = base in ("int8", "fp8")
    return dict(
        checkpoint=ck,
        act_offload=act,
        stream=st,
        stream_resident=2,
        grad_offload=go,
        optim_state=opt,
        base_precision=base if quant else "compute",
    )


def _valid(ck, act, st, go, acc, opt, base):
    _, full, _ = _setup(base, st)
    try:
        return resolve("balanced", _overrides(ck, act, st, go, opt, base), DEVICE, full)
    except ValueError as err:
        return err


COMBOS = list(_combos())
VALID = [c for c in COMBOS if not isinstance(_valid(*c), ValueError)]
INVALID = [c for c in COMBOS if isinstance(_valid(*c), ValueError)]
_REFS: dict = {}


def _ids(c):
    return "-".join(str(x) for x in c)


@pytest.mark.parametrize("combo", INVALID, ids=_ids)
def test_invalid_combination_rejected(combo):
    err = _valid(*combo)
    assert isinstance(err, ValueError) and "invalid memory plan" in str(err)


def _errors(combo, lr_scale: float = 1.0) -> tuple[float, float]:
    """(max relative loss error, max relative master-update error) vs the reference."""
    ck, act, st, go, acc, opt, base = combo
    dtype, full, quant = _setup(base, st)
    key = (base, full, acc)
    if key not in _REFS:
        losses, masters, _ = _reference(dtype, full, quant, base, acc)
        _REFS[key] = (losses, masters)
    ref_losses, ref_masters = _REFS[key]
    init = {
        n: p.detach().float().cpu() for n, p in _model(dtype, full).named_parameters()
    }
    model = _model(dtype, full)
    session = MemorySession(
        model,
        _valid(*combo),
        DEVICE,
        dtype,
        grad_acc=acc,
        trainable_base=full,
        lr=LR * lr_scale,
        quant_min_features=0,
    )
    losses = []
    for _ in range(STEPS):
        for micro, data in enumerate(_batches(dtype, acc)):
            loss = _loss(model, data)
            losses.append(loss.item())
            session.backward(loss, micro)
        session.step()
    loss_err = max(
        abs(g - w) / max(abs(w), 1e-6) for g, w in zip(losses, ref_losses, strict=True)
    )
    masters = session.masters(model)
    assert masters.keys() == ref_masters.keys()
    update_err = 0.0
    for name, want in ref_masters.items():
        delta_ref = want - init[name]
        delta = masters[name] - init[name]
        rel = (delta - delta_ref).norm() / (delta_ref.norm() + 1e-12)
        update_err = max(update_err, float(rel))
    return loss_err, update_err


@pytest.mark.parametrize("combo", VALID, ids=_ids)
def test_combination_matches_reference(combo):
    dtype = _setup(combo[-1], combo[2])[0]
    loss_err, update_err = _errors(combo)
    assert loss_err <= TOL[dtype] and update_err <= UPDATE_TOL[dtype], (
        loss_err,
        update_err,
    )


def test_matrix_check_catches_a_wrong_update():
    combo = ("block", True, "swap", True, 4, "stream", BASES[0])
    dtype = _setup(combo[-1], combo[2])[0]

    def check():
        loss_err, update_err = _errors(combo, lr_scale=1.25)
        assert loss_err <= TOL[dtype] and update_err <= UPDATE_TOL[dtype]

    must_fail(check, "a 25% larger learning rate")
