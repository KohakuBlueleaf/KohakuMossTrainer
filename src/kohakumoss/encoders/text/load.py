"""Streamed text-encoder load: the text model is built with its parameters on the meta
device and filled from the checkpoint's memory-mapped safetensors one module at a
time, each tensor straight to its home under the plan's placement: the compute
device, or the host as the checkpoint's own file pages (cast to the compute dtype by
the copy that moves them on). A module quantized at load goes through the work
device and lands at its home quantized; stored-quantized layers attach their stored
bases over the file pages. No dense copy of the model is made in host memory. See
docs/internals/text-encoder.md#streamed-load."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

import torch
from accelerate import init_empty_weights
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, PretrainedConfig

from kohakumoss.encoders.text.comfy_files import comfy_text_key, packaged_config
from kohakumoss.encoders.text.files import VALUES, TextFiles, hf_text_files
from kohakumoss.encoders.text.plan import DTYPES
from kohakumoss.logging.log import get_logger
from kohakumoss.offload.encoders.text import (
    QUANTIZE,
    TextPlacement,
    decode_w4a8,
    place_text,
    w4a8_extra_bytes,
)
from kohakumoss.offload.host.arena import arena_bytes, arena_kind
from kohakumoss.offload.weights.deferred import Deferred, placeholder
from kohakumoss.offload.weights.layers import device_dtype, module_tensors
from kohakumoss.quant.bind import work_device
from kohakumoss.quant.comfy_format import comfy_layers, layer_base
from kohakumoss.quant.convert import QuantLinear
from kohakumoss.quant.embedding import QuantEmbedding
from kohakumoss.quant.source import STORED, log_sources
from kohakumoss.quant.stored import SCALE
from kohakumoss.quant.stored_text import stored_base
from kohakumoss.quant.text import linear_fate, quantize_text_linears, quantizes
from kohakumoss.quant.weights import QuantWeight
from kohakumoss.types.plans import TextEncoderPlan
from kohakumoss_common.hostmem import host_fits
from kohakumoss_common.units import GIB

HOST = torch.device("cpu")
COMFY_VALUES = (torch.bfloat16, torch.float16)
DENSE_ROWS = 16384
log = get_logger(__name__)


@dataclass
class TextCheckpoint:
    """A text encoder's checkpoint: the text model's `config` (or a multimodal config
    holding it) and tensors (`files`); its stored-quantized layers by module path, kept
    as stored bases (`kept`: (() -> base over the file pages, bias name | None)) or
    dequantized into the dense weight (`dense`: () -> that base); the file names those
    layers own (`owned`); the stored dtypes a quantize-at-load source reads
    (`values`); `strict`: every other file tensor must be the model's."""

    name: str
    config: PretrainedConfig
    files: TextFiles
    kept: dict[str, tuple[Callable, str | None]] = field(default_factory=dict)
    dense: dict[str, Callable] = field(default_factory=dict)
    owned: set[str] = field(default_factory=set)
    values: tuple[torch.dtype, ...] = VALUES
    strict: bool = False


def hf_checkpoint(model_dir: Path) -> TextCheckpoint:
    """A model dir: its config and safetensors shards."""
    config = AutoConfig.from_pretrained(str(model_dir))
    return TextCheckpoint(str(model_dir), config, hf_text_files(model_dir))


def _under(key: str, path: str) -> bool:
    return key.startswith(path + ".")


def comfy_checkpoint(path: Path, keep: bool) -> TextCheckpoint:
    """A ComfyUI text-encoder file under the packaged config: its quantized layers kept
    as stored bases (`keep`) or dequantized."""
    files = TextFiles([Path(path)], comfy_text_key)
    config = packaged_config().text_config
    ck = TextCheckpoint(str(path), config, files, values=COMFY_VALUES, strict=True)
    for p, layer in comfy_layers(files.keys(), files.get, files.dtype).items():
        ck.owned.update(layer.keys)
        if keep:
            bias = f"{p}.bias" if f"{p}.bias" in files else None
            ck.kept[p] = (partial(layer_base, layer, files.get), bias)
            ck.owned.update(k for k in files.keys() if _under(k, p))
        else:
            ck.dense[p] = partial(layer_base, layer, files.get)
    return ck


def _same(key: str) -> str:
    return key


def stored_checkpoint(model_dir: Path, stored: Path, record: dict) -> TextCheckpoint:
    """A pre-quantized text-encoder file (`record` its text-quant record) under
    `model_dir`'s config."""
    files = TextFiles([Path(stored)], _same)
    config = AutoConfig.from_pretrained(str(model_dir))
    ck = TextCheckpoint(str(stored), config, files, strict=True)
    for p in record["layers"]:
        bias = f"{p}.bias" if f"{p}.bias" in files else None
        make = partial(_stored_base, record, files, p)
        ck.kept[p] = (make, bias)
        ck.owned.update(n for n in (f"{p}.weight", p + SCALE, bias) if n)
    return ck


def _stored_base(record: dict, files: TextFiles, path: str):
    return stored_base(record, files.get(f"{path}.weight"), files.get(path + SCALE))


def meta_text(config: PretrainedConfig, dtype: torch.dtype) -> nn.Module:
    """The text model of `config` (a multimodal config's language model) in `dtype`:
    parameters on the meta device, buffers built on the host."""
    with init_empty_weights(include_buffers=False):
        model = AutoModel.from_config(config, dtype=dtype)
    text = getattr(model, "language_model", model)
    return text.eval().requires_grad_(False)


def attach_base(model: nn.Module, path: str, base, bias, dtype: torch.dtype) -> None:
    """Replace module `path` (a linear or the embedding) by its stored base."""
    parent_path, _, name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    dense = getattr(parent, name)
    if isinstance(dense, nn.Embedding):
        quant = QuantEmbedding(base.q, base.scale, base.rotation, dtype)
    else:
        quant = QuantLinear.from_base(
            dense.in_features, dense.out_features, base, bias, dtype
        )
    setattr(parent, name, quant)


def dense_weight(base, dtype: torch.dtype, rows: int = DENSE_ROWS) -> Tensor:
    """`base.weight(dtype)`, a per-row QuantWeight built `rows` rows at a time (each
    row dequantizes and unrotates on its own, so the values are the same)."""
    if not isinstance(base, QuantWeight) or base.q.shape[0] <= rows:
        return base.weight(dtype)
    n = base.q.shape[0]
    out = torch.empty(base.q.shape, dtype=dtype)
    for i in range(0, n, rows):
        part = QuantWeight(base.q[i : i + rows], base.scale[i : i + rows], base.fmt, base.rotation)  # fmt: skip
        out[i : i + rows] = part.weight(dtype)
    return out


class _Fill:
    """The per-module fill of one load: `dtypes` (compute dtype by tensor id of a host
    file page stored in another), `views` (ids of tensors left as file pages),
    `deferred` (Deferred host forms by tensor id), `used` (file names read)."""

    def __init__(self, ck: TextCheckpoint, dtype: torch.dtype):
        self.ck = ck
        self.dtype = dtype
        self.dtypes: dict[int, torch.dtype] = {}
        self.views: set[int] = set()
        self.deferred: dict[int, Deferred] = {}
        self.used: set[str] = set()

    def dense_base(self, name: str):
        """The stored base dequantized into weight `name`, or None."""
        path = name.removesuffix(".weight")
        if name.endswith(".weight") and path in self.ck.dense:
            return self.ck.dense[path]()
        return None

    def value(self, name: str) -> tuple[Tensor, bool]:
        """(the values of tensor `name`, whether they are the file pages)."""
        base = self.dense_base(name)
        if base is not None:
            return dense_weight(base, self.dtype), False
        if name not in self.ck.files:
            raise ValueError(
                f"{self.ck.name}: the text model's {name!r} is not in the checkpoint"
            )
        self.used.add(name)
        return self.ck.files.get(name), True

    def __call__(
        self,
        module: nn.Module,
        prefix: str,
        home: torch.device | None,
        defer: bool = False,
    ) -> None:
        """Replace every meta parameter / buffer of `module` by its checkpoint values:
        a copy on `home` in its dtype, or (home None) the host value as read; `defer`
        keeps a dequantized weight's stored base as its Deferred host form."""
        for path, sub in module.named_modules():
            at = f"{prefix}{path}." if path else prefix
            for kind in (sub._parameters, sub._buffers):
                for name, t in list(kind.items()):
                    if t is None or t.device.type != "meta":
                        continue
                    base = self.dense_base(at + name) if defer else None
                    if base is not None:
                        kind[name] = self._deferred(t, base)
                    else:
                        kind[name] = self._made(t, *self.value(at + name), home)

    def _deferred(self, t: Tensor, base) -> Tensor:
        new = nn.Parameter(placeholder(t.shape, t.dtype), requires_grad=False)
        self.deferred[id(new)] = Deferred(base, t.shape)
        return new

    def _made(self, t: Tensor, src: Tensor, view: bool, home) -> Tensor:
        if home is not None:
            src = src.to(home, t.dtype)
        new = nn.Parameter(src, requires_grad=False) if isinstance(t, nn.Parameter) else src  # fmt: skip
        if home is None:
            if view:
                self.views.add(id(new))
            if src.dtype != t.dtype:
                self.dtypes[id(new)] = t.dtype
        return new

    def check(self) -> None:
        """Every file tensor read, owned by a stored layer, or the checkpoint is not
        strict."""
        if not self.ck.strict:
            return
        left = sorted(set(self.ck.files.keys()) - self.used - self.ck.owned)
        if left:
            raise ValueError(
                f"{self.ck.name} is not the Qwen3.5 text model: unexpected {left[:4]}"
            )


def _merge(report: dict, part: dict) -> None:
    for key, value in part.items():
        if isinstance(value, dict):
            into = report.setdefault(key, {})
            for k, v in value.items():
                into[k] = into.get(k, 0) + v
        elif isinstance(value, int):
            report[key] = report.get(key, 0) + value


def quantized_delta(text: nn.Module, plan: TextEncoderPlan, dtype: torch.dtype) -> int:
    """Bytes quantizing `text`'s plain linears adds (negative: removes), measured by
    quantizing one meta-device linear of each shape."""
    delta, seen = 0, {}
    for m in text.modules():
        if linear_fate(m, plan.precision, plan.quant_group) != "quantize":
            continue
        key = (m.in_features, m.out_features, m.bias is not None)
        if key not in seen:
            probe = nn.Sequential(nn.Linear(*key, device="meta", dtype=dtype))
            part = quantize_text_linears(
                probe,
                plan.precision,
                plan.quant_group,
                dtype,
                recipe=plan.fp8_recipe,
                quiet=True,
            )
            seen[key] = part["quant_bytes"] - part["dense_bytes"]
        delta += seen[key]
    return delta


def held_on_host(text: nn.Module, memory: str) -> list[Tensor]:
    """The tensors `memory`'s placement keeps on the host between passes and copies in
    (offload: all but the embedding; ring: the decoder layers')."""
    match memory:
        case "offload":
            emb = getattr(text, "embed_tokens", None)
            skip = (
                set() if emb is None else {id(t) for t in module_tensors(emb).values()}
            )
            return [t for t in module_tensors(text).values() if id(t) not in skip]
        case "ring":
            return [t for m in text.layers for t in module_tensors(m).values()]
    return []


def copy_bytes(held: list[Tensor], fill: _Fill, kind: str) -> int:
    """Host bytes `host="copy"` adds over the loaded form of `held`: the HostArena of
    their copies in the compute dtype (`kind`: `arena_kind`), less the private host
    tensors it replaces; a Deferred host form is not copied."""
    copied = [t for t in held if id(t) not in fill.deferred]
    sizes = [t.numel() * device_dtype(t, fill.dtypes).itemsize for t in copied]
    replaced = sum(t.nbytes for t in copied if id(t) not in fill.views)
    return arena_bytes(sizes, kind) - replaced


def _warn_dense(ck: TextCheckpoint, text: nn.Module, dtype: torch.dtype) -> None:
    """A WARNING when the dequantized layers an encoder running on the host holds do
    not fit the host's room."""
    need = sum(text.get_submodule(p).weight.numel() * dtype.itemsize for p in ck.dense)
    if not host_fits(need):
        log.warning(
            f"text encoder memory 'cpu': the {len(ck.dense)} dequantized layers of "
            f"{ck.name} take {need / GIB:.2f} GiB of host memory, more than the host "
            "has room for"
        )


@torch.no_grad()
def load_placed_text(
    ck: TextCheckpoint,
    plan: TextEncoderPlan,
    device: torch.device | str,
    choose: Callable,
    choose_w4a8: Callable,
) -> tuple[nn.Module, TextPlacement, dict, TextEncoderPlan]:
    """(text model, placement, quantization report, plan) of `ck` under `plan` on
    `device`: the W4A8 decode (`choose_w4a8(model, device)` for `auto`) and
    `memory="auto"` (`choose(model, device, extra_layers=bytes the load adds)`) are
    settled before any weight is read, the host form (`plan.host` `auto`: copy when
    it fits the host's room) after."""
    device = torch.device(device)
    dtype = DTYPES[plan.dtype]
    text = meta_text(ck.config, dtype)
    for p, (make, bias) in ck.kept.items():
        attach_base(
            text, p, make(), None if bias is None else ck.files.get(bias), dtype
        )
    quantize = plan.precision in QUANTIZE and not plan.stored
    work = work_device(device)
    decode = None
    if plan.precision == "w4a8":
        decode = plan.w4a8 if plan.w4a8 != "auto" else choose_w4a8(text, device)
        plan = replace(plan, w4a8=decode)
    if plan.memory == "auto":
        extra = w4a8_extra_bytes(text) if decode == "int8" else 0
        if quantize:
            extra += quantized_delta(text, plan, dtype)
        plan = replace(plan, memory=choose(text, device, extra_layers=extra))
    resident = plan.memory == "resident"
    streams = plan.memory in ("offload", "ring")
    if ck.dense and plan.memory == "cpu":
        _warn_dense(ck, text, dtype)
    report = {"stored": plan.stored} if plan.stored else {}
    if quantize:
        report.update(layers=0, dense_bytes=0, quant_bytes=0, skipped=0, sources={})
    fill = _Fill(ck, dtype)
    parts = [(f"layers.{i}.", m, True) for i, m in enumerate(text.layers)]
    parts += [(f"{n}.", m, False) for n, m in text.named_children() if n != "layers"]
    for prefix, module, layer in parts:
        on_device = resident or (
            plan.memory == "ring" and not layer and prefix != "embed_tokens."
        )
        home = device if on_device else None
        if quantize and quantizes(module, plan.precision, plan.quant_group):
            fill(module, prefix, work)
            part = quantize_text_linears(
                module,
                plan.precision,
                plan.quant_group,
                dtype,
                recipe=plan.fp8_recipe,
                work_device=work,
                source=ck.files.source(prefix, ck.values),
                quiet=True,
            )
            _merge(report, part)
            for t in module_tensors(module).values():
                t.data = t.data.to(home or HOST)
        else:
            fill(module, prefix, home, defer=layer and streams and home is None)
            if quantize:
                fates = (linear_fate(m, plan.precision, plan.quant_group) for m in module.modules())  # fmt: skip
                report["skipped"] += sum(f == "skip" for f in fates)
        if decode == "int8":
            n = decode_w4a8(module, work, home or HOST)
            report["w4a8_decoded"] = report.get("w4a8_decoded", 0) + n
    fill.check()
    if quantize:
        report["from_source"] = report["sources"].get(STORED, 0)
        log_sources(f"text encoder {plan.precision}", report["sources"])
    host = plan.host
    if host == "auto":
        kind = arena_kind(plan.pinned, device)
        need = copy_bytes(held_on_host(text, plan.memory), fill, kind)
        host = "copy" if host_fits(need) else "file"
    plan = replace(plan, host=host)
    placement = place_text(text, plan, device, dtype, host, fill.dtypes, fill.deferred)
    return text, placement, report, plan
