"""Resume equality on CUDA, bit-exact: the resume tests run in a child process with
KOHAKUMOSS_DETERMINISTIC=1 (set before the kernels are imported)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.cuda, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[2]
TESTS = (
    "tests/training/test_entry_smoke.py::test_lora_smoke_and_resume",
    "tests/training/test_online_cache.py::test_online_cache_resume_equals_uninterrupted",
)


def test_cuda_resume_is_bit_exact_in_deterministic_mode(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    env = {**os.environ, "KOHAKUMOSS_DETERMINISTIC": "1"}
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS]
    cmd += ["--basetemp", str(tmp_path / "child")]
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-6000:] + out.stderr[-3000:]
    assert f"{len(TESTS)} passed" in out.stdout, out.stdout[-3000:]
