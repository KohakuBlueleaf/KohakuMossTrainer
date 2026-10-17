"""Qwen3.5 text encoder wrapper: chat-template prompt -> unpadded last hidden states."""

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

MAX_LENGTH = 4096


def _subdir(path: Path, name: str) -> Path:
    return path / name if (path / name).is_dir() else path


def tokenizer_dir(path: Path) -> Path:
    """`<path>/tokenizer`, else the sibling `<path>/../tokenizer` of a model dir, else `path`."""
    for cand in (path / "tokenizer", path.parent / "tokenizer"):
        if cand.is_dir():
            return cand
    return path


class QwenTextEncoder:
    """Frozen encoder running on `device`; embeddings land on `output_device` (default
    `device`). `offload="cpu"` keeps weights on CPU and moves them per call."""

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device | str = "cuda",
        offload: str = "none",
        max_length: int = MAX_LENGTH,
        output_device: torch.device | str | None = None,
    ):
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.output_device = torch.device(output_device or device)
        self.max_length = max_length
        match offload:
            case "none":
                self.model.to(self.device)
                self._run = self._encode
            case "cpu":
                self.model.to("cpu")
                self._run = self._encode_offloaded
            case _:
                raise ValueError(f"unknown text-encoder offload {offload!r}")
        self._empty = self._run([""])[0]

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        offload: str = "none",
        max_length: int = MAX_LENGTH,
        output_device: torch.device | str | None = None,
    ) -> "QwenTextEncoder":
        """`path` is a pipeline root (text_encoder/ + tokenizer/) or a model dir (its
        sibling tokenizer/ is used when present)."""
        path = Path(path)
        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir(path)))
        model = AutoModel.from_pretrained(
            str(_subdir(path, "text_encoder")), dtype=dtype
        )
        return cls(model, tok, device, offload, max_length, output_device)

    def format(self, prompt: str) -> str:
        """The exact training-time chat template (thinking enabled, no system prompt)."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def token_ids(self, prompt: str) -> Tensor:
        ids = self.tokenizer(
            self.format(prompt),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        return ids["input_ids"]

    @torch.no_grad()
    def _encode(self, prompts: Sequence[str]) -> list[Tensor]:
        out = []
        for prompt in prompts:
            ids = self.token_ids(prompt).to(self.device)
            hidden = self.model(input_ids=ids, use_cache=False, return_dict=True)
            out.append(hidden.last_hidden_state[0].to(self.output_device))
        return out

    def _encode_offloaded(self, prompts: Sequence[str]) -> list[Tensor]:
        self.model.to(self.device)
        try:
            return self._encode(prompts)
        finally:
            self.model.to("cpu")

    def encode(self, prompts: Sequence[str]) -> list[Tensor]:
        """EncodeText: one [L_i, text_dim] tensor per prompt, each encoded alone (unpadded)."""
        return self._run(prompts)

    def __call__(self, prompts: Sequence[str]) -> list[Tensor]:
        return self.encode(prompts)

    def empty(self) -> Tensor:
        """Embedding of the empty prompt (CFG negative / text dropout), encoded at init."""
        return self._empty
