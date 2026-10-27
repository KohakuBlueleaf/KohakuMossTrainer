"""Text-encoder prewarm: record the kernel calls of one synthetic pack through a
weightless twin of the encoder's text model and replay them at every pack-shape bucket
the encoder's plan can produce (segment count, longest and mean segment, total tokens),
so live encodes bind every tuned choice from the tables. See
docs/internals/tuning.md#text-encoder-buckets.
"""

import copy
from types import MethodType

import torch
from torch import nn

from kohakumoss.encoders.text.forward import KernelForward, make_pack
from kohakumoss.model.build import materialize
from kohakumoss.offload.encoders.text import TextPlacement
from kohakumoss.quant.bind import rebind_quant
from kohakumoss.quant.convert import QuantLinear
from kohakumoss.quant.embedding import QuantEmbedding
from kohakumoss.quant.text import GroupQuantLinear
from kohakumoss_kernels.hardware.current import on_device
from kohakumoss_kernels.recording import recording
from kohakumoss_kernels.tuning.buckets import (
    LENGTH_FLOOR,
    MEAN_FLOOR,
    SEGMENT_CAP,
    TOKEN_FLOOR,
    delta_buckets,
    length_bucket,
    pow2,
    segment_bucket,
)
from kohakumoss_kernels.tuning.replay import OpReport, Target, replay

RECORD_ROWS = (37, 24)
RECORD_TOKEN = 1
QUANTIZED_MODULES = (QuantLinear, GroupQuantLinear, QuantEmbedding)


def longest_row(te, max_text_tokens: int, data_tokens: int = 0) -> int:
    """Longest token row an encode can see: the text cap plus the chat template,
    within the encoder's max length; with `data_tokens` (the data's longest caption
    estimate) at most the length bucket of that estimate plus the template."""
    template = len(te.token_ids([""])[0])
    cap = min(max_text_tokens + template, te.max_length)
    if data_tokens:
        cap = min(cap, length_bucket(data_tokens + template))
    return max(1, cap)


def _counts(bucket: int, batch: int) -> range:
    """Segment counts in segment bucket `bucket`, at most `batch`."""
    lo = 1 if bucket == 1 else bucket // 2 + 1
    hi = batch if bucket == SEGMENT_CAP else min(bucket, batch)
    return range(lo, hi + 1)


def _fits(lengths: tuple[int, ...], max_tokens: int, padded: bool) -> bool:
    cost = len(lengths) * max(lengths) if padded else sum(lengths)
    return not max_tokens or cost <= max_tokens


def _with_mean(longest: int, count: int, mean: int) -> tuple[int, ...] | None:
    """`count` rows, one of `longest` tokens, the others equal and as short as the
    mean bucket `mean` allows (fewest tokens), or None when no such pack exists."""
    if count == 1:
        return (longest,) if pow2(longest, MEAN_FLOOR) == mean else None
    low = 0 if mean == MEAN_FLOOR else mean // 2
    r = max(1, (count * (low + 1) - longest + count - 2) // (count - 1))
    if r > longest:
        return None
    lengths = (longest, *([r] * (count - 1)))
    return lengths if pow2(sum(lengths) // count, MEAN_FLOOR) == mean else None


def _longest(bucket: int, max_len: int) -> list[int]:
    """Longest-row lengths tried for length bucket `bucket`: its top (within
    `max_len`) and its bottom (fewest tokens)."""
    bottom = 1 if bucket == LENGTH_FLOOR else bucket // 2 + 1
    return list(dict.fromkeys(n for n in (min(bucket, max_len), bottom) if n >= 1))


def pack_lengths(
    batch: int, max_tokens: int, max_len: int, padded: bool = False
) -> list[tuple[int, ...]]:
    """Row lengths of synthetic packs covering every (segments, longest, mean) bucket of
    the delta rule and every token bucket of the tuned ops, for packs of at most
    `batch` rows and `max_tokens` tokens (0: no cap; padded packs count every row at
    the longest) with rows of at most `max_len` tokens."""
    packs: dict[tuple, tuple[int, ...]] = {}
    segment_buckets = sorted({segment_bucket(s) for s in range(1, batch + 1)})
    length = LENGTH_FLOOR
    while length <= length_bucket(max_len):
        for sb in segment_buckets:
            mean = MEAN_FLOOR
            while mean <= length:
                for longest in _longest(length, max_len):
                    for count in _counts(sb, batch):
                        lengths = (
                            (longest,) * count
                            if padded
                            else _with_mean(longest, count, mean)
                        )
                        if lengths is None or not _fits(lengths, max_tokens, padded):
                            continue
                        n = _total(lengths, padded)
                        key = delta_buckets(n, len(lengths), longest)
                        packs.setdefault(key, lengths)
                mean *= 2
        length *= 2
    tokens = {pow2(_total(p, padded), TOKEN_FLOOR) for p in packs.values()}
    cap = min(max_tokens or batch * max_len, batch * max_len)
    bucket = TOKEN_FLOOR
    while bucket <= pow2(cap, TOKEN_FLOOR):
        n = min(bucket, cap)
        count = -(-n // max_len)
        if bucket not in tokens and count <= batch:
            rest = n - (count - 1) * max_len
            lengths = (max_len,) * (count - 1) + (rest,)
            if padded:
                lengths = (max_len,) * count
            if _fits(lengths, max_tokens, padded):
                packs[("tokens", bucket)] = lengths
        bucket *= 2
    return list(dict.fromkeys(packs.values()))


def _total(lengths: tuple[int, ...], padded: bool) -> int:
    return len(lengths) * max(lengths) if padded else sum(lengths)


def text_targets(
    te, max_text_tokens: int, recorded: int, data_tokens: int = 0
) -> list[Target]:
    """Replay targets of a pack recorded with `recorded` tokens: one per bucket pack."""
    heads_v = getattr(te.model.config, "linear_num_value_heads", 1)
    targets = []
    max_len = longest_row(te, max_text_tokens, data_tokens)
    for lengths in pack_lengths(te.batch, te.plan.max_tokens, max_len, te.padded):
        n = _total(lengths, te.padded)
        seg = (max(lengths),) * len(lengths) if te.padded else lengths
        rows = {recorded: n, recorded * heads_v: n * heads_v}
        targets.append(Target(rows, {recorded: seg}))
    return targets


def meta_copy(module: nn.Module) -> nn.Module:
    """A copy of `module` whose every tensor is an empty meta tensor of the same shape
    and dtype; bound kernels and other callables are shared, not copied (the twin is
    re-bound by `rebind_quant`)."""
    memo = {}
    for m in module.modules():
        for t in m._parameters.values():
            if t is not None:
                memo[id(t)] = nn.Parameter(torch.empty_like(t, device="meta"), False)
        for t in m._buffers.values():
            if t is not None:
                memo[id(t)] = torch.empty_like(t, device="meta")
        for value in vars(m).values():
            shared = callable(value) and not isinstance(value, MethodType)
            if shared and not isinstance(value, (nn.Module, torch.Tensor)):
                memo[id(value)] = value
    return copy.deepcopy(module, memo)


def mirror_quantized(twin: nn.Module, model: nn.Module) -> int:
    """Put a meta copy of every quantized module of `model` (QuantLinear,
    GroupQuantLinear, QuantEmbedding: quantized at load or stored in the file) at the
    same path of `twin`; returns how many."""
    count = 0
    for path, m in model.named_modules():
        if not isinstance(m, QUANTIZED_MODULES):
            continue
        parent_path, _, name = path.rpartition(".")
        parent = twin.get_submodule(parent_path) if parent_path else twin
        setattr(parent, name, meta_copy(m))
        count += 1
    return count


def text_shapes(te) -> nn.Module:
    """A weightless twin of `te`'s text model: its class built on the meta device,
    `te`'s quantized modules mirrored (`mirror_quantized`), every tensor in the dtype
    of `te`'s tensor of the same name as a stride-0 view on `te.device`, quantized
    GEMMs bound there."""
    with torch.device("meta"):
        twin = type(te.model)(te.model.config)
    twin.eval().requires_grad_(False)
    mirror_quantized(twin, te.model)
    like = {**dict(te.model.named_parameters()), **dict(te.model.named_buffers())}
    materialize(twin, te.device, like)
    rebind_quant(twin, te.device, te.dtype, autograd=False)
    return twin


def record_text(te) -> tuple[list, int]:
    """The kernel calls of one synthetic pack through `te`'s weightless twin (on
    `te`'s ops, no real weight read), and the pack's token count."""
    twin = text_shapes(te)
    layers = list(getattr(twin, "layers", []))
    place = TextPlacement(twin, layers, getattr(twin, "embed_tokens", None), te.device)
    forward = KernelForward(twin, place, te.forward.ops)
    rows = [[RECORD_TOKEN] * n for n in RECORD_ROWS]
    pack = make_pack(rows, te.device, te.pad_id, te.padded)
    with recording() as calls, torch.no_grad():
        forward([pack])
    return calls, int(pack.cu[-1])


def prewarm_text(te, max_text_tokens: int, data_tokens: int = 0) -> dict[str, OpReport]:
    """Tune the text-encoder ops at every pack bucket of `te`'s plan, rows up to
    `longest_row(te, max_text_tokens, data_tokens)` (a no-op for an encoder without
    the kernel forward)."""
    if not isinstance(getattr(te, "forward", None), KernelForward):
        return {}
    with on_device(torch.device(te.device)):
        calls, recorded = record_text(te)
        targets = text_targets(te, max_text_tokens, recorded, data_tokens)
        return replay(calls, targets, backward=False)
