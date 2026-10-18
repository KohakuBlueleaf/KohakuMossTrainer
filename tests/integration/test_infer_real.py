"""Real inference: the pipeline on the real weights (text encoder streamed, the DiT
under `balanced`, tuning from the stored table). Generates are finite, deterministic
per seed and sampled without autograd; an adapter (bypass) changes the images and is
removed exactly; the prompt template is the official one; the phase messages come in
order."""

import numpy as np
import pytest
import torch

from kohakumoss import adapters as adapter_api
from kohakumoss.inference import MossPipeline
from kohakumoss.model.numerics import output_row_scales
from tests._util.logs import assert_phases, collecting
from tests._util.negative import must_fail

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
PROMPTS = [
    "a red brick house beside a lake",
    "a lighthouse, rocky coast, sunset, clouds",
]
GEN = dict(sizes=(512, 512), steps=8, cfg=4.0, seeds=[3, 9], batch=2)
TEMPLATE = "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n"
PHASES = [
    "encoding 2 prompt(s) ...",
    "encoding 2 prompt(s): done",
    "sampling 2 image(s)",
    "decoding 2 image(s) ...",
    "decoding 2 image(s): done",
    "generate: 2 image(s)",
]


@pytest.fixture(scope="module")
def pipe(weights_root):
    return MossPipeline.from_pretrained(
        weights_root / "base",
        dtype=torch.bfloat16,
        device="cuda",
        adapter_mode="bypass",
        vae_plan={"memory": "offload"},
        text_plan={"memory": "stream"},
        memory="balanced",
        tune="table",
    )


def pixels(result) -> list[np.ndarray]:
    return [np.asarray(im, dtype=np.int16) for im in result.images]


def check_images(arrays: list[np.ndarray]) -> None:
    for a in arrays:
        assert a.shape == (512, 512, 3) and a.std() > 2.0, a.std()


def same(a, b) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_generate_is_deterministic_and_untracked(pipe):
    grads = []
    with collecting() as logs:
        first = pipe.generate(
            PROMPTS, step_callback=lambda i, x: grads.append(x.requires_grad), **GEN
        )
    check_images(pixels(first))
    assert grads and not any(grads)
    assert same(pixels(first), pixels(pipe.generate(PROMPTS, **GEN)))
    assert_phases(logs.messages, PHASES)
    must_fail(lambda: check_images([np.zeros((512, 512, 3))]), "a blank image")
    must_fail(lambda: assert_phases(logs.messages, PHASES[::-1]), "phases reversed")


def test_adapter_changes_and_is_removed(pipe, adapter):
    base = pixels(pipe.generate(PROMPTS, **GEN))
    stack = adapter_api.load(
        pipe.backbone,
        adapter,
        1.0,
        mode="bypass",
        row_scales=output_row_scales(pipe.backbone),
    )
    pipe.backbone.requires_grad_(False)
    pipe.backbone.bind_ops()
    try:
        tuned = pixels(pipe.generate(PROMPTS, **GEN))
    finally:
        stack.remove()
        pipe.backbone.bind_ops()
    check_images(tuned)
    assert not same(tuned, base), "the adapter changed nothing"
    assert same(pixels(pipe.generate(PROMPTS, **GEN)), base), "not removed exactly"


def test_prompt_template(pipe):
    te = pipe.place.text_encoder
    assert te.format("a cat") == TEMPLATE
    want = list(te.tokenizer(TEMPLATE)["input_ids"])
    assert te.token_ids(["a cat"])[0] == want
    must_fail(lambda: _equal(te.format("a dog"), TEMPLATE), "another prompt")


def _equal(a, b) -> None:
    assert a == b
