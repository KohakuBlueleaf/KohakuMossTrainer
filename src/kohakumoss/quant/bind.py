"""Binding quantized linears to their compute device and dtype after a placement."""

import torch
from torch import nn

from kohakumoss.quant.compute import ComputeWeight
from kohakumoss.quant.embedding import QuantEmbedding
from kohakumoss.quant.w4a8 import W4A8Weight
from kohakumoss.quant.weights import QuantWeight

BOUND = (QuantWeight, ComputeWeight, W4A8Weight, QuantEmbedding)


def rebind_quant(module: nn.Module, device, dtype, autograd: bool = True) -> None:
    """Bind every quantized base to (device, dtype); `autograd=False` binds the
    fp8-compute bases of a model that never runs backward to the gradient-free op."""
    for m in module.modules():
        if isinstance(m, ComputeWeight):
            m.bind(dtype, device, autograd=autograd)
        elif isinstance(m, BOUND):
            m.bind(dtype, device)


def set_int8_compute(module: nn.Module, dgrad: str) -> int:
    """Select int8 compute (the `int8_linear` op, input gradient `dgrad`) on every int8
    base inside `module` (int8 QuantWeights and int8-decoding W4A8 bases); they bind it
    on their next `bind`. Returns how many."""
    count = 0
    for m in module.modules():
        int8 = isinstance(m, QuantWeight) and m.fmt == "int8"
        if int8 or (isinstance(m, W4A8Weight) and m.decodes_to_int8):
            m.int8_dgrad = dgrad
            count += 1
    return count


def work_device(device) -> torch.device:
    """Where weights are quantized: the compute device when it is an accelerator."""
    device = torch.device(device)
    return device if device.type in ("cuda", "xpu") else torch.device("cpu")
