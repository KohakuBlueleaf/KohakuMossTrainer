"""The streaming checkpoint writer: files the safetensors (Rust) reader loads exactly,
and a save that holds about one tensor of extra host memory (peak RSS of a fresh
process), where copying the state to the host first holds a second whole copy."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from kohakumoss.training.checkpoint import stream
from kohakumoss.training.checkpoint.stream import save_stream
from tests._util.negative import must_fail
from tests.training.checkpoint._checkpoint_peak import COUNT, NUMEL

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[3]
CHILD = "tests.training.checkpoint._checkpoint_peak"
STATE_BYTES = COUNT * NUMEL * 4


def _extra_peak(tmp_path, writer: str) -> int:
    out = subprocess.run(
        [sys.executable, "-m", CHILD, writer, str(tmp_path / f"{writer}.safetensors")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr
    return int(out.stdout.split()[-1])


def _below_half_the_state(tmp_path, writer: str) -> None:
    extra = _extra_peak(tmp_path, writer)
    assert extra < STATE_BYTES // 2, (writer, extra, STATE_BYTES)


@pytest.mark.budget(30)  # two child processes
def test_save_holds_no_second_copy_of_the_state(tmp_path):
    _below_half_the_state(tmp_path, "stream")
    must_fail(
        lambda: _below_half_the_state(tmp_path, "whole"),
        "the whole state copied to the host before the write",
    )


def _state() -> dict:
    g = torch.Generator().manual_seed(0)
    wide = torch.randn(6, 10, generator=g)
    return {
        "w32": torch.randn(5, 7, generator=g),
        "w16": torch.randn(3, 4, generator=g).half(),
        "wbf": torch.randn(9, generator=g).bfloat16(),
        "strided": wide.t()[2:5],
        "scalar": torch.tensor(3.5),
        "empty": torch.zeros(0, 4),
        "step": torch.tensor([7], dtype=torch.int64),
        "mask": torch.tensor([True, False, True]),
    }


def _loads_exactly(path, state: dict, meta: dict) -> None:
    back = load_file(str(path))
    assert back.keys() == state.keys()
    for k, v in state.items():
        assert back[k].dtype == v.dtype and back[k].shape == v.shape, k
        assert torch.equal(back[k], v), k
    with safe_open(str(path), "pt") as f:
        assert f.metadata() == meta


def test_written_file_loads_exactly_with_the_reference_reader(tmp_path, monkeypatch):
    state, meta = _state(), {"kohakumoss_step": "7"}
    save_stream(state, tmp_path / "a.safetensors", meta)
    _loads_exactly(tmp_path / "a.safetensors", state, meta)
    real = stream.host_bytes
    monkeypatch.setattr(stream, "host_bytes", lambda t: real(torch.zeros_like(t)))
    save_stream(state, tmp_path / "b.safetensors", meta)
    must_fail(
        lambda: _loads_exactly(tmp_path / "b.safetensors", state, meta),
        "tensor bytes not the tensor's own",
    )
