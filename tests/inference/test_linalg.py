"""An owned CUDA pipeline runs generate under cuSOLVER (OFT's batched Cayley inverse)
and leaves the process-wide linalg backend as it found it."""

import pytest
import torch

from kohakumoss.inference import MossPipeline
from kohakumoss.model import random_backbone
from tests._util.negative import must_fail

pytestmark = pytest.mark.cuda

CUSOLVER = torch._C._LinalgBackend.Cusolver


def _backend():
    return torch.backends.cuda.preferred_linalg_library()


def _generate(pipe, seen):
    embeds = [torch.randn(4, pipe.backbone.cfg.text_dim, device="cuda")]
    pipe.generate(
        prompt_embeds=embeds,
        sizes=(32, 32),
        steps=2,
        cfg=1.0,
        output="latent",
        step_callback=lambda i, x: seen.append(_backend()),
    )


def _check(seen, after, before):
    assert seen and all(b == CUSOLVER for b in seen)
    assert after == before


def test_owned_cuda_pipeline_scopes_cusolver():
    torch.backends.cuda.preferred_linalg_library("default")
    before = _backend()
    model = random_backbone("tiny", device="cuda", dtype=torch.float32)
    shared = MossPipeline.from_components(model, device="cuda")
    shared_seen = []
    _generate(shared, shared_seen)
    must_fail(lambda: _check(shared_seen, _backend(), before), "no cusolver")
    pipe = MossPipeline(model, None, None, device="cuda", owned=True, tune="off")
    assert _backend() == before
    seen = []
    _generate(pipe, seen)
    _check(seen, _backend(), before)
    torch.backends.cuda.preferred_linalg_library("cusolver")
    leaked = _backend()
    torch.backends.cuda.preferred_linalg_library("default")
    must_fail(lambda: _check(seen, leaked, before), "backend leaked")
    with pipe._linalg():
        q = torch.randn(8, 16, 16, device="cuda")
        eye = torch.eye(16, device="cuda")
        q = q - q.transpose(-1, -2)
        inv = torch.linalg.inv(eye - q)
    assert torch.allclose(inv @ (eye - q), eye.expand(8, -1, -1), atol=1e-4)
