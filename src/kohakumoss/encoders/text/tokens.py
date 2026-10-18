"""Prompt -> token ids (the training chat template) and grouping of prompts into
encoder calls."""

from collections.abc import Sequence
from pathlib import Path

import torch

MAX_LENGTH = 4096
TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json")


def tokenizer_dir(path: Path) -> Path:
    """`path` itself when it holds tokenizer files, else `<path>/tokenizer`, else the
    sibling `<path>/../tokenizer` of a model dir, else `path`."""
    if any((path / name).is_file() for name in TOKENIZER_FILES):
        return path
    for cand in (path / "tokenizer", path.parent / "tokenizer"):
        if cand.is_dir():
            return cand
    return path


def chat_text(tokenizer, prompt: str) -> str:
    """The exact training-time chat template (thinking enabled, no system prompt)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def token_ids(tokenizer, prompts: Sequence[str], max_length: int) -> list[list[int]]:
    """Token ids of every prompt under the chat template, truncated to `max_length`."""
    out = []
    for prompt in prompts:
        ids = tokenizer(
            chat_text(tokenizer, prompt),
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        out.append(torch.as_tensor(ids).flatten().tolist())
    return out


def plan_groups(
    lengths: Sequence[int], batch: int, max_tokens: int, padded: bool
) -> list[list[int]]:
    """Prompt indices per encoder call, longest first: at most `batch` prompts and
    `max_tokens` tokens per call (padded calls count every row at the longest length;
    0 = no token cap). A prompt longer than the cap gets a call of its own."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    groups: list[list[int]] = []
    cur: list[int] = []
    cost = 0
    for i in order:
        n = lengths[i]
        new = (len(cur) + 1) * lengths[cur[0]] if padded and cur else cost + n
        if cur and (len(cur) >= batch or (max_tokens and new > max_tokens)):
            groups.append(cur)
            cur, cost = [], 0
            new = n
        cur.append(i)
        cost = new
    if cur:
        groups.append(cur)
    return groups
