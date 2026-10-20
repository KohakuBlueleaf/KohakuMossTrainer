"""Load-time placement lines name where each component's weights live and where they
compute, with sizes: byte splits, every text-encoder placement, a resident and an
offloaded VAE (model-free: the modules are tensor holders on chosen devices; the DiT
lines run in tests/integration)."""

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from kohakumoss.offload.encoders import report
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
GIB = 2**30


def _names(line: str, *parts: str) -> None:
    """The line carries every placement word and size."""
    for part in parts:
        assert part in line, (part, line)


def _equal(a, b) -> None:
    assert a == b


def _holder(**tensors: torch.Tensor) -> nn.Module:
    """A module that only owns the given tensors: floating ones as parameters, the
    rest as buffers."""
    m = nn.Module()
    for name, t in tensors.items():
        if t.is_floating_point():
            m.register_parameter(name, nn.Parameter(t, requires_grad=False))
        else:
            m.register_buffer(name, t)
    return m


def test_split_bytes_and_sizes():
    m = _holder(
        w_dev=torch.empty(4, 4, dtype=torch.bfloat16, device="meta"),
        b_dev=torch.empty(4, dtype=torch.bfloat16, device="meta"),
        w_host=torch.empty(4, 4, dtype=torch.bfloat16),
        idx_host=torch.zeros(3, dtype=torch.int64),
    )
    on, host = 4 * 4 * 2 + 4 * 2, 4 * 4 * 2 + 3 * 8
    assert report.split_bytes(m) == (on, host)
    assert report.split_bytes(m, "cpu") == (on + host, 0)
    assert report.split_bytes(None) == (0, 0)
    assert report.gib_text(3 * GIB / 2) == "1.50 GiB"
    must_fail(lambda: _equal(report.split_bytes(m), (on + host, 0)), "all on device")
    must_fail(lambda: _equal(report.split_bytes(m), (on, 4 * 4 * 2)), "buffers dropped")


@dataclass
class Plan:
    precision: str = "fp8"
    dtype: str = "bf16"
    free_after_use: bool = False


class Placement:
    def __init__(self, name, device, slots=2, res=GIB // 4, run=GIB):
        self.name, self.device, self.slots = name, torch.device(device), slots
        self.pinned = name in ("ring", "offload")
        self.host = "copy"
        self._res, self._run = res, run

    def resident_bytes(self):
        return self._res

    def pass_bytes(self):
        return self._run


class TextEncoder:
    def __init__(self, placement, model, plan=None):
        self.placement = placement
        self.plan = plan or Plan()
        self.model = model


def _host_half_gib() -> nn.Module:
    """Half a GiB of host weights (uninitialised, never touched)."""
    return _holder(w=torch.empty(GIB // 2, dtype=torch.uint8))


@pytest.mark.parametrize(
    "name, words",
    [
        (
            "ring",
            [
                "-> ring (weights in host RAM, pinned, 0.50 GiB;",
                "ring of 2 layers on cuda:0",
            ],
        ),
        (
            "offload",
            [
                "-> offload (weights in host RAM, pinned, 0.50 GiB;",
                "moved to cuda:0 per pass",
            ],
        ),
        ("resident", ["-> resident on cuda:0 (0.25 GiB; precision fp8, bf16)"]),
        (
            "cpu",
            [
                "-> cpu (weights in host RAM, 0.50 GiB;",
                f"runs on the host, {torch.get_num_threads()} threads",
            ],
        ),
    ],
)
def test_text_encoder_lines(name, words):
    device = "cpu" if name == "cpu" else "cuda:0"
    te = TextEncoder(Placement(name, device), _host_half_gib())
    line, fields = report.te_line(te, "w/text_encoder", "cuda:0")
    _names(line, "text encoder: w/text_encoder -> ", "precision fp8", *words)
    assert fields["gpu_pass_gib"] == 1.0 and fields["gpu_gib"] == 0.25
    assert fields["host_gib"] == 0.5
    if name == "ring":
        _names(line, "~0.75 GiB, 0.25 GiB resident")
    old = "loading text encoder w/text_encoder on cuda:0"
    must_fail(lambda: _names(old, *words), "the old load line")
    on_device = _holder(w=torch.empty(GIB // 2, dtype=torch.uint8, device="meta"))
    moved, _ = report.te_line(
        TextEncoder(Placement(name, device), on_device), "p", "cuda:0"
    )
    if name != "resident":
        must_fail(lambda: _names(moved, "0.50 GiB;"), "device weights read as host RAM")


def test_vae_lines():
    vae = _holder(w=torch.empty(GIB // 4, dtype=torch.uint8))
    line, fields = report.vae_line(vae, "w/vae", "offload", "cuda:0")
    _names(line, "VAE: w/vae -> offload (host RAM, moved to cuda:0 per pass, 0.25 GiB)")
    assert fields == {"gib": 0.25}
    line, _ = report.vae_line(vae, "w/vae", "ring", "cuda:0")
    _names(line, "VAE: w/vae -> ring (host RAM, 0.25 GiB;", "2-layer ring on cuda:0")
    line, _ = report.vae_line(vae, "w/vae", "cpu", "cpu")
    threads = torch.get_num_threads()
    _names(
        line,
        f"VAE: w/vae -> cpu (host RAM, 0.25 GiB; runs on the host, {threads} threads)",
    )
    line, _ = report.vae_line(vae, "w/vae", "gpu", "cuda:0")
    _names(line, "VAE: w/vae -> resident on cuda:0 (0.25 GiB)")
    must_fail(lambda: _names(line, "host RAM"), "a resident VAE read as offloaded")
    empty, _ = report.vae_line(_holder(), "w/vae", "gpu", "cuda:0")
    must_fail(lambda: _names(empty, "0.25 GiB"), "the size of another VAE")
