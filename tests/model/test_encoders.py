"""Encoders: latent patch layout against an explicit index map; prompt template."""

import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from kohakumoss.encoders.vae import patchify, unpatchify


def _patchify_loop(z):
    b, c, hh, ww = z.shape
    out = torch.empty(b, c * 4, hh // 2, ww // 2)
    for ch in range(c):
        for i in range(2):
            for j in range(2):
                out[:, ch * 4 + i * 2 + j] = z[:, ch, i::2, j::2]
    return out


@pytest.mark.cpu
def test_patch_layout():
    z = torch.randn(2, 3, 6, 8)
    p = patchify(z)
    assert torch.equal(p, _patchify_loop(z))
    assert torch.equal(unpatchify(p), z)
    swapped = z.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not torch.equal(patchify(swapped.flip(-1)), _patchify_loop(z))


@pytest.mark.real_weights
def test_prompt_template():
    root = Path(os.environ["KOHAKUMOSS_WEIGHTS"]) / "base"
    tok = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    text = tok.apply_chat_template(
        [{"role": "user", "content": "a cat"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    assert text == "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n"
