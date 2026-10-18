"""QwenTextEncoder: prompts -> unpadded last hidden states, under a TextEncoderPlan."""

import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

from kohakumoss.encoders.specs import text_spec
from kohakumoss.encoders.text.forward import (
    KernelForward,
    TextOps,
    make_pack,
    op_static,
)
from kohakumoss.encoders.text.hf import HfForward
from kohakumoss.encoders.text.plan import (
    DTYPES,
    check_plan,
    resolve_dtype,
    text_plan,
)
from kohakumoss.encoders.text.tokens import (
    MAX_LENGTH,
    chat_text,
    plan_groups,
    token_ids,
    tokenizer_dir,
)
from kohakumoss.logging.log import bar as log_bar
from kohakumoss.offload.accel import device_budget
from kohakumoss.offload.encoders.text import apply_to_text_encoder
from kohakumoss.offload.encoders.text_memory import choose_text_memory, text_sizes
from kohakumoss.quant.stored_text import load_stored_text, read_text_record
from kohakumoss.types.plans import TextEncoderPlan
from kohakumoss_common.calls import nothing

ENCODER_ID = "qwen3.5-9b"


@contextmanager
def hf_bars_off():
    """transformers' own progress bars off inside the block (the caller logs the load)."""
    was = hf_logging.is_progress_bar_enabled()
    hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        if was:
            hf_logging.enable_progress_bar()


def _subdir(path: Path, name: str) -> Path:
    return path / name if (path / name).is_dir() else path


def default_auto_memory(
    text, device: torch.device, plan: TextEncoderPlan, reserve: int = 0
) -> str:
    """The fastest placement whose footprint fits the device's free memory (minus
    `reserve` bytes kept for the caller's own peak); the host has no limit."""
    if device.type == "cpu":
        return "resident"
    sizes = text_sizes(text, plan.stream_resident)
    budget = int(device_budget(device))
    tokens = plan.max_tokens or 16384
    choice, _ = choose_text_memory(sizes, budget, reserve, tokens, tokens)
    return choice


def _stored(model_dir: Path, plan: TextEncoderPlan):
    """The model of `model_dir` with the plan's pre-quantized file attached."""
    record = read_text_record(plan.stored)
    if record is None or record["precision"] != plan.precision:
        found = None if record is None else record["precision"]
        raise ValueError(
            f"text-encoder stored file {plan.stored} holds precision {found!r}, the "
            f"plan asks for {plan.precision!r}"
        )
    return load_stored_text(model_dir, plan.stored, DTYPES[plan.dtype])


def _released(*args):
    raise RuntimeError(
        "the text encoder was freed after its first pass (free_after_use)"
    )


class QwenTextEncoder:
    """Frozen Qwen3.5 encoder placed by `plan` (memory mode, precision, batching,
    layout); embeddings land on `output_device` (default `device`). The empty-prompt
    embedding is encoded with the first pass. `stats` accumulates passes, prompts,
    tokens, seconds and host-to-device weight bytes. `progress = True` shows a bar per
    pass (layers or packs) at INFO level. `encoder_spec` is the compute spec a text
    cache key records."""

    encoder_id = ENCODER_ID

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device | str = "cuda",
        plan: TextEncoderPlan | dict | None = None,
        max_length: int = MAX_LENGTH,
        output_device: torch.device | str | None = None,
        auto_memory: Callable | None = None,
    ):
        plan = text_plan(plan)
        compute = torch.device("cpu") if plan.memory == "cpu" else torch.device(device)
        self.plan = check_plan(plan, device=compute)
        if hasattr(model, "visual"):
            del model.visual
        text = getattr(model, "language_model", model)
        self.model = text.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = compute
        self.output_device = torch.device(output_device or device)
        self.dtype = next(text.parameters()).dtype
        choose = auto_memory or partial(default_auto_memory, plan=plan)
        self.placement, self.quant, plan = apply_to_text_encoder(
            text, plan, compute, self.dtype, choose
        )
        self.plan = plan
        self.encoder_spec = text_spec(plan, self.dtype)
        self.pad_id = getattr(tokenizer, "pad_token_id", None) or 0
        self.padded = plan.layout == "padded"
        self.batch = 1 if plan.layout == "single" else plan.batch
        if plan.forward == "kernels":
            static = op_static(text.config)
            ops = TextOps(compute, self.dtype, plan.kernels, static)
            self.forward = KernelForward(text, self.placement, ops)
            self.op_names = ops.names
        else:
            self.forward = HfForward(text, self.placement, self.pad_id)
            self.op_names = {}
        self._empty: Tensor | None = None
        self.stats = {"passes": 0, "prompts": 0, "tokens": 0, "seconds": 0.0}
        self.flops_per_token = 2 * sum(
            p.numel() for n, p in text.named_parameters() if "embed_tokens" not in n
        ) + 2 * sum(b.numel() for n, b in text.named_buffers() if n.endswith(".q"))
        self.before_pass = self.after_pass = nothing
        self.progress = False
        self._run = self._encode

    def set_timeline(self, timeline) -> None:
        """Mark `step` (the work since the last pass) before and `te` after each pass."""
        self.before_pass = partial(timeline.mark, "step")
        self.after_pass = partial(timeline.mark, "te")

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: torch.device | str = "cuda",
        plan: TextEncoderPlan | dict | None = None,
        max_length: int = MAX_LENGTH,
        output_device: torch.device | str | None = None,
        auto_memory: Callable | None = None,
    ) -> "QwenTextEncoder":
        """`path` is a pipeline root (text_encoder/ + tokenizer/) or a model dir (its own
        tokenizer files, else a sibling tokenizer/). Loaded on the host in the plan's
        dtype, then placed; `auto_memory(text_model, device) -> memory mode` resolves
        `memory="auto"` (default: the fastest placement that fits the free memory)."""
        path = Path(path)
        plan = text_plan(plan)
        on = torch.device("cpu") if plan.memory == "cpu" else torch.device(device)
        plan = check_plan(resolve_dtype(plan, on), device=on)
        model_dir = _subdir(path, "text_encoder")
        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir(model_dir)))
        with hf_bars_off():
            model = (
                _stored(model_dir, plan)
                if plan.stored
                else AutoModel.from_pretrained(str(model_dir), dtype=DTYPES[plan.dtype])
            )
        return cls(model, tok, device, plan, max_length, output_device, auto_memory)

    def format(self, prompt: str) -> str:
        """The exact training-time chat template (thinking enabled, no system prompt)."""
        return chat_text(self.tokenizer, prompt)

    def token_ids(self, prompts: Sequence[str]) -> list[list[int]]:
        return token_ids(self.tokenizer, prompts, self.max_length)

    @torch.no_grad()
    def _encode(self, prompts: Sequence[str]) -> list[Tensor]:
        start = time.perf_counter()
        want_empty = self._empty is None
        texts = list(prompts) + ([""] if want_empty else [])
        if not texts:
            return []
        rows = self.token_ids(texts)
        lengths = [len(r) for r in rows]
        groups = plan_groups(lengths, self.batch, self.plan.max_tokens, self.padded)
        packs = [
            make_pack([rows[i] for i in g], self.device, self.pad_id, self.padded)
            for g in groups
        ]
        self.before_pass()
        self.placement.begin()
        pbar = log_bar(
            total=self.forward.units(packs),
            desc="text encode",
            unit=self.forward.unit,
            enabled=self.progress,
            leave=False,
        )
        pbar.set_postfix_str(f"{len(packs)} packs, {sum(lengths)} tokens")
        try:
            with pbar:
                hidden = self.forward(packs, partial(pbar.update, 1))
            out: list[Tensor | None] = [None] * len(texts)
            for g, p, h in zip(groups, packs, hidden):
                for i, part in zip(g, h.split(p.lengths, 0)):
                    out[i] = part.to(self.output_device)
            del hidden
        finally:
            self.placement.end()
        self.after_pass()
        if want_empty:
            self._empty = out.pop()
        for dev in {self.device, self.output_device}:
            if dev.type in ("cuda", "xpu"):
                torch.accelerator.synchronize(dev)
        self.stats["passes"] += 1
        self.stats["prompts"] += len(prompts)
        self.stats["tokens"] += sum(lengths)
        self.stats["seconds"] += time.perf_counter() - start
        self.stats["h2d_bytes"] = self.placement.h2d_bytes
        if self.plan.free_after_use:
            self.release()
        return out

    def release(self) -> None:
        """Free every weight (host and device); later encodes raise."""
        self.placement.release()
        self.model = None
        self.forward = None
        self._run = _released
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()

    def before_first(self, fn: Callable[[], None]) -> None:
        """Run `fn()` once, right before the next pass (the kernel prewarm of a run
        that encodes online)."""
        run = self._run

        def first(prompts: Sequence[str]) -> list[Tensor]:
            self._run = run
            fn()
            return run(prompts)

        self._run = first

    def encode(self, prompts: Sequence[str]) -> list[Tensor]:
        """EncodeText: one [L_i, text_dim] tensor per prompt (unpadded)."""
        return self._run(prompts)

    def __call__(self, prompts: Sequence[str]) -> list[Tensor]:
        return self.encode(prompts)

    def empty(self) -> Tensor:
        """Embedding of the empty prompt (CFG negative / text dropout)."""
        if self._empty is None:
            self._run([])
        return self._empty

    def footprint(self) -> dict[str, int]:
        """Device bytes between passes and during one (weights only)."""
        return {
            "resident_bytes": self.placement.resident_bytes(),
            "pass_bytes": self.placement.pass_bytes(),
        }
