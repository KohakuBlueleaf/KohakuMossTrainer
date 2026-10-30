"""The text encoder's end-of-pass synchronize (`output_to`): none when the weights stay
resident and the embeddings stay on the compute device, else the devices the pass
touched; and TEXT_ENCODE ahead_device 'auto' (`ahead_device`)."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.encoders.text.encoder import QwenTextEncoder
from kohakumoss.training.build.model import ahead_device
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(5)]

CUDA = torch.device("cuda", 0)
CPU = torch.device("cpu")


def _enc(memory: str, device=CUDA):
    return SimpleNamespace(plan=SimpleNamespace(memory=memory), device=device)


def _check(output_to) -> None:
    resident = _enc("resident")
    output_to(resident, CUDA)
    assert resident.sync_devices == () and resident.output_device == CUDA
    to_host = _enc("resident")
    output_to(to_host, CPU)
    assert set(to_host.sync_devices) == {CUDA, CPU}
    streamed = _enc("ring")
    output_to(streamed, CUDA)
    assert streamed.sync_devices == (CUDA,)
    on_cpu = _enc("cpu", CPU)
    output_to(on_cpu, CPU)
    assert on_cpu.sync_devices == (CPU,)


def test_output_to():
    _check(QwenTextEncoder.output_to)


def test_output_to_negative():
    def never_syncs(enc, device):
        enc.output_device = torch.device(device)
        enc.sync_devices = ()

    must_fail(lambda: _check(never_syncs), "sync dropped for host / streamed outputs")


def test_ahead_device_auto():
    assert ahead_device(_enc("resident"), CUDA) == CUDA
    assert ahead_device(_enc("ring"), CUDA) == CPU
    assert ahead_device(_enc("resident", torch.device("cuda", 1)), CUDA) == CPU
