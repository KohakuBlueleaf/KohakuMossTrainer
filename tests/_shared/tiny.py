"""Shared helpers: a synthetic model, a batch, a loss+grads snapshot, device params."""

import pytest
import torch

from kohakumoss.training.memory.synthetic import SyntheticModel, synthetic_batch

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.cuda),
    pytest.param("xpu", marks=pytest.mark.xpu),
]
HW = [(3, 4), (2, 5)]
TEXT = [5, 3]


def make_model(device="cpu", dtype=torch.float64, seed=0, depth=4, dim=32, ffn=64):
    torch.manual_seed(seed)
    return SyntheticModel(dim, ffn, depth, 8).to(device, dtype)


def batch(device="cpu", dtype=torch.float64, seed=1):
    return synthetic_batch(HW, TEXT, 8, device, dtype, seed)


def loss_and_grads(model, data, freeze_blocks: bool = False):
    img, txt, sigma, seq, target = data
    if freeze_blocks:
        for p in model.layers.parameters():
            p.requires_grad_(False)
    pred = model(img, txt, sigma, seq)
    loss = (pred.double() - target.double()).square().mean()
    loss.backward()
    grads = {
        n: p.grad.detach().double().cpu().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return loss.detach().double().cpu(), grads


def max_diff(a: dict, b: dict) -> float:
    assert a.keys() == b.keys(), (sorted(a), sorted(b))
    return max((a[k] - b[k]).abs().max().item() for k in a)
