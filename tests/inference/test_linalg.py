"""An owned CUDA pipeline selects cuSOLVER for linalg (OFT's batched Cayley inverse)."""

import pytest
import torch

from kohakumoss.inference import MossPipeline
from kohakumoss.model import random_backbone

pytestmark = pytest.mark.cuda


def test_owned_cuda_pipeline_prefers_cusolver():
    torch.backends.cuda.preferred_linalg_library("default")
    model = random_backbone("tiny", device="cuda", dtype=torch.float32)
    MossPipeline.from_components(model, device="cuda")
    assert (
        torch.backends.cuda.preferred_linalg_library()
        != torch._C._LinalgBackend.Cusolver
    )
    MossPipeline(model, None, None, device="cuda", owned=True, tune="off")
    assert (
        torch.backends.cuda.preferred_linalg_library()
        == torch._C._LinalgBackend.Cusolver
    )
    q = torch.randn(8, 16, 16, device="cuda")
    eye = torch.eye(16, device="cuda")
    q = q - q.transpose(-1, -2)
    inv = torch.linalg.inv(eye - q)
    assert torch.allclose(inv @ (eye - q), eye.expand(8, -1, -1), atol=1e-4)
