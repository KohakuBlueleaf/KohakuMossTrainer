"""Per-group hyperparameters survive every optimiser path.

Each GPU / host-state / streamed / host AdamW, alone and inside a MemorySession, must
reproduce `torch.optim.AdamW` run with the same groups (different lr, weight decay,
betas, eps per group). The check must fail when the groups collapse into one.
"""

import pytest
import torch

from kohakumoss.offload.cpu_optim import CPUAdamW
from kohakumoss.offload.fused_step import PerParamOptimizer
from kohakumoss.offload.gpu_adamw import HostStateAdamW, MasterAdamW
from kohakumoss.training.memory.profiles import resolve
from kohakumoss.training.memory.session import MemorySession
from kohakumoss.training.memory.synthetic import SyntheticModel, synthetic_batch
from tests._util.negative import must_fail

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
HP = [
    dict(lr=1e-2, weight_decay=0.1),
    dict(lr=2e-2, weight_decay=0.0, betas=(0.8, 0.95)),
    dict(lr=5e-3, weight_decay=0.05, eps=1e-6),
]
STEPS = 3


def _collapse(groups: list[dict]) -> list[dict]:
    """One group with the largest weight decay (the failure mode under test)."""
    wd = max(g.get("weight_decay", 0.01) for g in groups)
    params = [p for g in groups for p in g["params"]]
    return [{"params": params, "lr": groups[0]["lr"], "weight_decay": wd}]


def _tensors(device):
    torch.manual_seed(0)
    shapes = [(16, 8), (8,), (12, 4), (4,), (6, 6)]
    return [torch.randn(s, device=device) for s in shapes]


def _grads(device):
    g = torch.Generator().manual_seed(1)
    shapes = [(16, 8), (8,), (12, 4), (4,), (6, 6)]
    return [
        [torch.randn(s, generator=g).to(device) for s in shapes] for _ in range(STEPS)
    ]


def _groups(params) -> list[dict]:
    return [
        {"params": [params[0], params[4]], **HP[0]},
        {"params": [params[1], params[3]], **HP[1]},
        {"params": [params[2]], **HP[2]},
    ]


def _run(device, make, collapse=False, per_param=False):
    params = [torch.nn.Parameter(t) for t in _tensors(device)]
    groups = _groups(params)
    opt = make(_collapse(groups) if collapse else groups)
    for grads in _grads(device):
        for p, g in zip(params, grads):
            p.grad = g.clone()
        if per_param:
            for p in reversed(params):
                opt.step_param(p)
            if hasattr(opt, "finish"):
                opt.finish()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
    if hasattr(opt, "synchronize"):
        opt.synchronize()
    return [p.detach().float().cpu() for p in params]


def _err(got, want) -> float:
    return max((a - b).abs().max().item() for a, b in zip(got, want))


OPTIMS = {
    "master": MasterAdamW,
    "host_state": HostStateAdamW,
    "cpu_adamw": CPUAdamW,
    "per_param": lambda groups: PerParamOptimizer(groups, torch.optim.AdamW),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("per_param", [False, True])
@pytest.mark.parametrize("name", list(OPTIMS))
def test_optimizer_keeps_group_hyperparameters(device, name, per_param):
    want = _run(device, torch.optim.AdamW)
    got = _run(device, OPTIMS[name], per_param=per_param)
    assert _err(got, want) < 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_group_check_catches_collapsed_groups(device):
    want = _run(device, torch.optim.AdamW)

    def check():
        assert _err(_run(device, MasterAdamW, collapse=True), want) < 1e-5

    must_fail(check, "groups collapsed into one (largest weight decay)")


def _model(device):
    torch.manual_seed(0)
    return SyntheticModel(32, 64, 4, 8).to(device, torch.float32)


def _model_groups(model) -> list[dict]:
    """MOSS-like rules: attention / FFN block weights, other weights, 1-D params."""
    named = dict(model.named_parameters())
    attn = [p for n, p in named.items() if ".q." in n or ".o." in n]
    ffn = [p for n, p in named.items() if "ffn" in n]
    taken = {id(p) for p in attn + ffn}
    rest = [p for p in named.values() if id(p) not in taken]
    return [
        {"params": attn, **HP[0]},
        {"params": ffn, **HP[1]},
        {"params": [p for p in rest if p.ndim > 1], **HP[2]},
        {"params": [p for p in rest if p.ndim <= 1], "lr": 3e-2, "weight_decay": 0.0},
    ]


def _batches(device, acc):
    return [
        synthetic_batch([(3, 4), (2, 3)], [4, 2], 8, device, seed=10 + i)
        for i in range(acc)
    ]


def _loss(model, data):
    img, txt, sigma, seq, target = data
    return (model(img, txt, sigma, seq) - target).square().mean()


def _reference(device, acc):
    model = _model(device)
    init = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(_model_groups(model))
    for _ in range(STEPS):
        for data in _batches(device, acc):
            _loss(model, data).backward()
        opt.step()
        opt.zero_grad()
    return init, {n: p.detach().cpu() for n, p in model.named_parameters()}


SESSION = [
    (stream, optim, go, acc)
    for stream in ("off", "swap")
    for optim in ("gpu", "stream", "cpu")
    for go in (False, True)
    for acc in (1, 2)
]


def _session_err(device, combo, collapse=False) -> float:
    stream, optim, go, acc = combo
    init, want = _reference(device, acc)
    model = _model(device)
    plan = resolve(
        "balanced",
        dict(
            checkpoint="block",
            stream=stream,
            stream_resident=2,
            optim_state=optim,
            grad_offload=go,
        ),
        device,
        trainable_base=True,
    )
    groups = _model_groups(model)
    session = MemorySession(
        model,
        plan,
        device,
        torch.float32,
        grad_acc=acc,
        trainable_base=True,
        param_groups=_collapse(groups) if collapse else groups,
    )
    for _ in range(STEPS):
        for micro, data in enumerate(_batches(device, acc)):
            session.backward(_loss(model, data), micro)
        session.step()
    got = session.masters(model)
    assert got.keys() == want.keys()
    err = 0.0
    for name, w in want.items():
        delta_ref = w - init[name]
        rel = (got[name] - init[name] - delta_ref).norm() / (delta_ref.norm() + 1e-12)
        err = max(err, float(rel))
    return err


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("combo", SESSION, ids=lambda c: "-".join(map(str, c)))
def test_session_keeps_group_hyperparameters(device, combo):
    assert _session_err(device, combo) < 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_session_group_check_catches_collapsed_groups(device):
    def check():
        assert _session_err(device, ("swap", "stream", True, 2), collapse=True) < 1e-3

    must_fail(check, "session groups collapsed into one")


@pytest.mark.cpu
def test_session_rejects_an_ungrouped_trainable_parameter():
    model = _model("cpu")
    plan = resolve("balanced", dict(checkpoint="block"), "cpu", trainable_base=True)
    groups = _model_groups(model)[:3]
    with pytest.raises(ValueError, match="no param group"):
        MemorySession(model, plan, "cpu", torch.float32, param_groups=groups)


@pytest.mark.cpu
def test_fused_per_param_groups_track_schedule_edits():
    params = [torch.nn.Parameter(t) for t in _tensors("cpu")]
    per = PerParamOptimizer(_groups(params), torch.optim.SGD, lr=1.0)
    per.param_groups[2]["lr"] = 0.0
    before = params[2].detach().clone()
    params[2].grad = torch.ones_like(params[2])
    per.step_param(params[2])
    assert torch.equal(params[2].detach(), before)
