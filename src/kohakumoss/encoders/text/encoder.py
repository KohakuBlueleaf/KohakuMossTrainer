"""QwenTextEncoder: prompts -> unpadded last hidden states, under a TextEncoderPlan."""

import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from kohakumoss.encoders.specs import text_spec
from kohakumoss.encoders.text.comfy_files import is_comfy_text_file, packaged_tokenizer
from kohakumoss.encoders.text.comfy_load import text_file_format
from kohakumoss.encoders.text.forward import (
    KernelForward,
    TextOps,
    make_pack,
    op_static,
)
from kohakumoss.encoders.text.hf import HfForward
from kohakumoss.encoders.text.load import (
    comfy_checkpoint,
    hf_checkpoint,
    load_placed_text,
    stored_checkpoint,
)
from kohakumoss.encoders.text.plan import (
    DTYPES,
    check_plan,
    resolve_dtype,
    resolve_precision,
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
from kohakumoss.logging.log import get_logger, hl
from kohakumoss.offload.accel import device_budget, release_host_cache
from kohakumoss.offload.encoders.text import apply_to_text_encoder, w4a8_extra_bytes
from kohakumoss.offload.encoders.text_memory import choose_text_memory, text_sizes
from kohakumoss.quant.convert import QuantLinear
from kohakumoss.quant.source import SourceWeights
from kohakumoss.quant.stored_text import read_text_record
from kohakumoss.quant.text import GroupQuantLinear
from kohakumoss.types.plans import TextEncoderPlan
from kohakumoss.weights import TEXT, comfy_file, hf_root, is_comfy_folder
from kohakumoss_common.calls import nothing
from kohakumoss_kernels.dtypes import short
from kohakumoss_kernels.hardware.current import on_device, synchronize

ENCODER_ID = "qwen3.5-9b"
log = get_logger(__name__)


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
    text,
    device: torch.device,
    plan: TextEncoderPlan,
    reserve: int = 0,
    extra_layers: int = 0,
) -> str:
    """The fastest placement whose footprint fits the device's free memory (minus
    `reserve` bytes kept for the caller's own peak) once the load adds `extra_layers`
    bytes to the decoder layers; 'cpu' on the host."""
    if device.type == "cpu":
        return "cpu"
    sizes = text_sizes(text, plan.stream_resident, extra_layers)
    budget = int(device_budget(device))
    tokens = plan.max_tokens or 16384
    choice, _ = choose_text_memory(sizes, budget, reserve, tokens, tokens)
    return choice


def _stored(model_dir: Path, plan: TextEncoderPlan):
    """The checkpoint of the plan's pre-quantized file under `model_dir`'s config."""
    record = read_text_record(plan.stored)
    if record is None or record["precision"] != plan.precision:
        found = None if record is None else record["precision"]
        raise ValueError(
            f"text-encoder stored file {plan.stored} holds precision {found!r}, the "
            f"plan asks for {plan.precision!r}"
        )
    return stored_checkpoint(model_dir, Path(plan.stored), record)


def text_source_of(path: str | Path) -> Path:
    """The text-encoder weights `path` names: a ComfyUI folder's default text-encoder
    file, an HF weights root's `base/` pipeline (`hf_root`), else `path` (a ComfyUI
    file, a pipeline root or a model dir)."""
    if not str(path):
        return Path(path)
    path = Path(path)
    return comfy_file(path, TEXT) if is_comfy_folder(path) else hf_root(path, TEXT)


def stored_text_format(path: str | Path, plan: TextEncoderPlan) -> tuple[str, str]:
    """(precision, layout) the text-encoder weights at `path` store (headers only): a
    ComfyUI file's (`text_file_format`), a plan's `stored` file's record precision, else
    `compute`."""
    path = text_source_of(path)
    if is_comfy_text_file(path):
        return text_file_format(path)
    if plan.stored:
        record = read_text_record(plan.stored)
        if record is not None:
            return record["precision"], ""
    return "compute", ""


def resolve_text_plan(
    path: str | Path,
    plan: TextEncoderPlan,
    device: torch.device | str,
    run_device: torch.device | str | None = None,
) -> tuple[TextEncoderPlan, str, str]:
    """(plan with precision resolved for `device`, where the encoder computes, and
    dtype for `run_device` (default `device`), the device its embeddings are for, and
    `device`; the stored precision; the stored layout) of the text encoder at `path`."""
    stored, layout = stored_text_format(path, plan)
    plan = resolve_precision(plan, stored, device)
    return resolve_dtype(plan, run_device or device, device), stored, layout


def default_w4a8(
    text, device: torch.device, plan: TextEncoderPlan, reserve: int = 0
) -> str:
    """`w4a8="auto"`: `int8` (decode once) when the int8-decoded encoder fits resident
    next to `reserve` bytes on `device` (the host: always), else `packed`."""
    if device.type == "cpu":
        return "int8"
    extra = w4a8_extra_bytes(text)
    sizes = text_sizes(text, plan.stream_resident, extra)
    budget = int(device_budget(device))
    tokens = plan.max_tokens or 16384
    choice, _ = choose_text_memory(sizes, budget, reserve, tokens, tokens)
    return "int8" if choice == "resident" else "packed"


def flops_per_token(text) -> int:
    """2 x the multiply-adds per token: every dense parameter outside the embedding,
    plus every quantized linear's [N, K]."""
    dense = sum(
        p.numel() for n, p in text.named_parameters() if "embed_tokens" not in n
    )
    quant = 0
    for m in text.modules():
        if isinstance(m, QuantLinear):
            quant += m.in_features * m.out_features
        elif isinstance(m, GroupQuantLinear):
            quant += m.q.numel()
    return 2 * (dense + quant)


def _released(*args):
    raise RuntimeError(
        "the text encoder was freed after its first pass (free_after_use)"
    )


def _first_pass(
    run: Callable, fn: Callable[[], None], encoder, prompts, limits=None
) -> list:
    """`before_first`'s pass: put `run` back, run `fn()`, then `run` the prompts."""
    encoder._run = run
    fn()
    return run(encoder, prompts, limits)


def cut_rows(rows: list[list[int]], limits) -> list[list[int]]:
    """Token ids of prompt i cut to its first `limits[i]` (0 or no limits: all)."""
    if not limits:
        return rows
    return [r[: limit or None] for r, limit in zip(rows, limits)]


class QwenTextEncoder:
    """Frozen Qwen3.5 encoder placed by `plan` (memory mode, precision, batching,
    layout); embeddings land on `output_device` (default `device`). The empty-prompt
    embedding is encoded with the first pass. `stats` accumulates passes, prompts,
    tokens, seconds and host-to-device weight bytes. `progress = True` shows a bar per
    pass (layers or packs) at INFO level. `encoder_spec` is the compute spec a text
    cache key records. `placed` (placement, quantization report, plan) is a model
    the streamed load already placed in compute dtype `dtype`."""

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
        source: SourceWeights | None = None,
        layout: str = "",
        auto_w4a8: Callable | None = None,
        placed: tuple | None = None,
        dtype: torch.dtype | None = None,
    ):
        plan = text_plan(plan)
        compute = torch.device("cpu") if plan.memory == "cpu" else torch.device(device)
        if plan.precision == "auto":
            plan = resolve_precision(plan, "compute", compute)
        self.plan = check_plan(plan, device=compute)
        if hasattr(model, "visual"):
            del model.visual
        text = getattr(model, "language_model", model)
        self.model = text.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = compute
        self.output_device = torch.device(output_device or device)
        self.dtype = dtype or text.norm.weight.dtype
        if placed is None:
            choose = auto_memory or partial(default_auto_memory, plan=plan)
            w4a8 = auto_w4a8 or partial(default_w4a8, plan=plan)
            placed = apply_to_text_encoder(
                text, plan, compute, self.dtype, choose, source, w4a8
            )
        self.placement, self.quant, plan = placed
        self.plan = plan
        self.layout = layout
        self.encoder_spec = text_spec(plan, self.dtype, layout)
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
        self.flops_per_token = flops_per_token(text)
        self.before_pass = self.after_pass = nothing
        self.progress = False
        self._run = QwenTextEncoder._encode

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
        auto_w4a8: Callable | None = None,
        run_device: torch.device | str | None = None,
    ) -> "QwenTextEncoder":
        """`path` is a pipeline root (text_encoder/ + tokenizer/), a model dir (its own
        tokenizer files, else a sibling tokenizer/), a ComfyUI text-encoder file (the
        packaged config and tokenizer) or a ComfyUI models folder (its default text
        encoder file). Streamed from the memory-mapped checkpoint in the plan's dtype
        at the plan's precision (`auto`: the stored one; a pre-quantized file's bases
        kept, or dequantized for another precision; quantized from the checkpoint's
        stored values) straight to each tensor's placed home (`load_placed_text`);
        `auto_memory(text_model, device, extra_layers=...) -> memory mode` resolves
        `memory="auto"` (default: the fastest placement that fits the free memory),
        `auto_w4a8(text_model, device)` a W4A8 file's `w4a8="auto"`; `dtype="auto"`
        resolves for `run_device` (default `device`), whatever the placement."""
        path = text_source_of(path)
        plan = text_plan(plan)
        on = torch.device("cpu") if plan.memory == "cpu" else torch.device(device)
        plan, stored, layout = resolve_text_plan(path, plan, on, run_device or device)
        plan = check_plan(plan, device=on)
        dtype = DTYPES[plan.dtype]
        if is_comfy_text_file(path):
            tok = packaged_tokenizer()
            keep = plan.precision == stored and stored != "compute"
            ck = comfy_checkpoint(path, keep)
            if stored != "compute" and not keep:
                log.info(
                    f"text encoder precision {hl(plan.precision)}: the {hl(stored)} "
                    f"layers of {hl(path, 'path')} dequantized to {short(dtype)}"
                )
        else:
            model_dir = _subdir(path, "text_encoder")
            tok = AutoTokenizer.from_pretrained(str(tokenizer_dir(model_dir)))
            ck = _stored(model_dir, plan) if plan.stored else hf_checkpoint(model_dir)
        choose = auto_memory or partial(default_auto_memory, plan=plan)
        w4a8 = auto_w4a8 or partial(default_w4a8, plan=plan)
        with hf_bars_off(), on_device(on):
            text, placement, report, plan = load_placed_text(ck, plan, on, choose, w4a8)
        return cls(
            text,
            tok,
            device,
            plan,
            max_length,
            output_device,
            layout=layout,
            placed=(placement, report, plan),
            dtype=dtype,
        )

    def format(self, prompt: str) -> str:
        """The exact training-time chat template (thinking enabled, no system prompt)."""
        return chat_text(self.tokenizer, prompt)

    def token_ids(self, prompts: Sequence[str]) -> list[list[int]]:
        return token_ids(self.tokenizer, prompts, self.max_length)

    @torch.no_grad()
    def _encode(self, prompts: Sequence[str], limits=None) -> list[Tensor]:
        """The prompts' embeddings (prompt i's first `limits[i]` tokens when given);
        the process's first encode also encodes the empty prompt, in a pass of its
        own, so the prompts' packs are the same on every call."""
        if self._empty is None:
            self._empty = self._pass([""])[0]
        out = self._pass(list(prompts), limits) if prompts else []
        self.stats["prompts"] += len(prompts)
        if self.plan.free_after_use:
            self.release()
        return out

    def _pass(self, texts: list[str], limits=None) -> list[Tensor]:
        """One text-encoder pass over `texts` (packed, placed, synchronised), text i
        cut to its first `limits[i]` tokens when given."""
        start = time.perf_counter()
        rows = cut_rows(self.token_ids(texts), limits)
        lengths = [len(r) for r in rows]
        groups = plan_groups(lengths, self.batch, self.plan.max_tokens, self.padded)
        packs = [
            make_pack([rows[i] for i in g], self.device, self.pad_id, self.padded)
            for g in groups
        ]
        self.before_pass()
        with on_device(torch.device(self.device)):
            out = self._placed_pass(texts, groups, packs, lengths)
        self.after_pass()
        for dev in {self.device, self.output_device}:
            synchronize(dev)
        self.stats["passes"] += 1
        self.stats["tokens"] += sum(lengths)
        self.stats["seconds"] += time.perf_counter() - start
        self.stats["h2d_bytes"] = self.placement.h2d_bytes
        return out

    def _placed_pass(self, texts, groups, packs, lengths) -> list[Tensor]:
        """The packs through the placed model (the caller holds the encoder's card
        current)."""
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
        return out

    def release(self) -> None:
        """Free every weight (host and device, the cached pinned host blocks too);
        later encodes raise."""
        self.placement.release()
        self.model = None
        self.forward = None
        self._run = _released
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()
        release_host_cache()

    def before_first(self, fn: Callable[[], None]) -> None:
        """Run `fn()` once, right before the next pass (the kernel prewarm of a run
        that encodes online)."""
        self._run = partial(_first_pass, self._run, fn)

    def encode(self, prompts: Sequence[str], limits=None) -> list[Tensor]:
        """EncodeText: one [L_i, text_dim] tensor per prompt (unpadded); with `limits`,
        prompt i is encoded from its first `limits[i]` tokens only (0: all). See
        docs/internals/text-encoder.md#over-long-prompts."""
        return self._run(self, prompts, limits)

    def __call__(self, prompts: Sequence[str]) -> list[Tensor]:
        return self.encode(prompts)

    def empty(self) -> Tensor:
        """Embedding of the empty prompt (CFG negative / text dropout)."""
        if self._empty is None:
            self._run(self, [])
        return self._empty

    def footprint(self) -> dict[str, int]:
        """Device bytes between passes and during one (weights only)."""
        return {
            "resident_bytes": self.placement.resident_bytes(),
            "pass_bytes": self.placement.pass_bytes(),
        }
