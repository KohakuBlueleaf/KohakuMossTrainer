"""The Lightning driver keeps the memory plan's placement: its LightningModule has no
parameter of its own, so Lightning's device moves (fit start, teardown) leave the
backbone where the plan put it (model-free: a one-parameter holder)."""

import pytest
import torch
from torch import nn

from kohakumoss.training.drivers.lightning import MossModule
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


class _Holder(nn.Module):
    """Owns one parameter: what a device move would reach."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4))


def _module(backbone) -> MossModule:
    return MossModule(
        backbone, None, None, None, None, None, None, None, None, None, (True,), None
    )


def check_kept(module: MossModule, backbone: nn.Module) -> None:
    assert list(module.parameters()) == [], "the module owns parameters"
    module.to("meta")
    assert backbone.weight.device.type == "cpu", "a device move reached the backbone"


def test_device_moves_leave_the_backbone():
    backbone = _Holder()
    check_kept(_module(backbone), backbone)
    registered = _Holder()
    old = _module(registered)
    nn.Module.__setattr__(old, "backbone", registered)
    must_fail(lambda: check_kept(old, registered), "backbone as a submodule")
    assert _module(backbone).held[0] is backbone
