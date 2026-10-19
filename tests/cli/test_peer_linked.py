"""The starter picks FSDP2 only when every pair of cards has peer access; PCIe cards
without P2P, or a machine without the cards, get DDP (model-free: a stand-in cuda)."""

from types import SimpleNamespace

import pytest

from kohakumoss.cli.config import starter
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _cuda(n: int, peer) -> SimpleNamespace:
    return SimpleNamespace(
        is_available=lambda: n > 0,
        device_count=lambda: n,
        can_device_access_peer=peer,
    )


def check(linked) -> None:
    assert linked(4, _cuda(4, lambda i, j: True))
    assert not linked(4, _cuda(4, lambda i, j: False))
    assert not linked(4, _cuda(4, lambda i, j: {i, j} != {0, 3}))
    assert not linked(4, _cuda(2, lambda i, j: True))
    assert not linked(4, _cuda(0, lambda i, j: True))


def test_fsdp_needs_peer_links():
    check(starter.peer_linked)
    must_fail(lambda: check(lambda gpus, cuda: True), "FSDP2 on any multi-GPU host")
