"""Where a frozen text encoder's weights live between and during its passes.

`resident`: everything on the device. `cpu`: everything on the host, run there.
`offload`: host (pinned) copies; every pass copies the whole model in and drops the
device copies after it. `ring`: decoder layers through a `LayerStreamer` ring, the
small rest resident. Under offload / ring the token embedding table stays on the
host and only the looked-up rows cross the bus. The host-held weights are copies
(`host="copy"`) or stay as they are (`host="file"`: the checkpoint's file pages);
`dtypes` maps a tensor id to its compute dtype where its host form holds another,
cast by the copy that moves it. `begin()` / `end()` bracket one pass; `end()` releases
every device weight buffer the pass used. See docs/internals/offload.md.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss.offload.accel import streamed
from kohakumoss.offload.encoders.placement import HostCopies
from kohakumoss.offload.weights.deferred import Deferred
from kohakumoss.offload.weights.layers import (
    LayerStreamer,
    device_dtype,
    module_bytes,
    module_tensors,
    tensor_bytes,
    tensor_ids,
)
from kohakumoss.quant.bind import rebind_quant, work_device
from kohakumoss.quant.compute import COMPUTE
from kohakumoss.quant.convert import QuantLinear
from kohakumoss.quant.embedding import QuantEmbedding, dense_lookup
from kohakumoss.quant.source import SourceWeights
from kohakumoss.quant.text import quantize_text_linears
from kohakumoss.quant.w4a8 import W4A8Weight
from kohakumoss.types.plans import TextEncoderPlan
from kohakumoss_common.calls import nothing


class TextPlacement:
    """Resident on `device` (the base; the other placements override the hooks)."""

    name = "resident"

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[nn.Module],
        embedding: nn.Embedding | None,
        device: torch.device | str,
        pinned: bool = True,
        host: str = "copy",
        dtypes: Mapping[int, torch.dtype] | None = None,
        deferred: Mapping[int, Deferred] | None = None,
    ):
        self.model = model
        self.layers = list(layers)
        self.embedding = embedding
        self.emb_ids = frozenset() if embedding is None else tensor_ids([embedding])
        self.dtypes = dict(dtypes or {})
        self.deferred = dict(deferred or {})
        self.lookup = host_lookup(embedding, self.dtypes)
        self.device = torch.device(device)
        self.pinned = pinned and streamed(self.device)
        self.host = host
        self.h2d_bytes = 0
        self.enter: list[Callable[[], None]] = [nothing] * len(self.layers)
        self.leave: list[Callable[[], None]] = [nothing] * len(self.layers)
        self.place()

    def to_device(self, t: Tensor) -> None:
        """Point `t` at its copy on the compute device, in its compute dtype."""
        src = self.deferred.get(id(t), t.data)
        t.data = src.to(self.device, device_dtype(t, self.dtypes))

    def place(self) -> None:
        for t in module_tensors(self.model).values():
            self.to_device(t)

    def embed(self, ids: Tensor) -> Tensor:
        """Embeddings of int64 token ids (any device) on the compute device."""
        return self.embedding(ids.to(self.device))

    def begin(self) -> None:
        return None

    def end(self) -> None:
        return None

    def resident_bytes(self) -> int:
        """Device bytes held between passes."""
        if self.device.type == "cpu":
            return 0
        return module_bytes(self.model)

    def pass_bytes(self) -> int:
        """Device weight bytes held during a pass."""
        return self.resident_bytes()

    def release(self) -> None:
        """Drop every weight (host and device); the placement is unusable after."""
        for t in module_tensors(self.model).values():
            t.data = torch.empty(0, dtype=t.dtype)
        self.layers = []
        self.lookup = None
        self.deferred = {}


class HostPlacement(TextPlacement):
    """Everything on the host; the encoder runs there (a tensor already on the host in
    its compute dtype is used as it is)."""

    name = "cpu"

    def __init__(self, model, layers, embedding, device="cpu", pinned=False, **kw):
        super().__init__(model, layers, embedding, "cpu", False, **kw)


class OffloadPlacement(TextPlacement):
    """Host copies; each pass moves the whole model (bar the embedding) in."""

    name = "offload"

    def place(self) -> None:
        tensors = list(module_tensors(self.model).values())
        moved = [t for t in tensors if id(t) not in self.emb_ids]
        self.copies = HostCopies(
            moved, self.device, self.pinned, self.host, self.dtypes, self.deferred
        )

    def embed(self, ids: Tensor) -> Tensor:
        return self.lookup(ids, self.device)

    def begin(self) -> None:
        self.copies.begin()
        self.h2d_bytes = self.copies.h2d_bytes

    def end(self) -> None:
        self.copies.end()

    def resident_bytes(self) -> int:
        return 0

    def pass_bytes(self) -> int:
        return self.copies.nbytes

    def release(self) -> None:
        self.copies.release()
        super().release()


class RingPlacement(TextPlacement):
    """Decoder layers streamed through `slots` ring slots per layer signature; the
    final norm and rotary tables resident; the embedding table on the host."""

    name = "ring"

    def __init__(self, model, layers, embedding, device, pinned=True, slots=2, **kw):
        self.slots = max(int(slots), 2)
        super().__init__(model, layers, embedding, device, pinned, **kw)

    def place(self) -> None:
        inside = tensor_ids(self.layers)
        for t in module_tensors(self.model).values():
            if id(t) not in inside and id(t) not in self.emb_ids:
                self.to_device(t)
        self.streamer = LayerStreamer(
            self.layers,
            self.device,
            slots=self.slots,
            pinned=self.pinned,
            host=self.host,
            dtypes=self.dtypes,
            deferred=self.deferred,
        )
        n = len(self.layers)
        self.enter = [partial(self.streamer.enter, i) for i in range(n)]
        self.leave = [partial(self.streamer.leave, i) for i in range(n)]

    def embed(self, ids: Tensor) -> Tensor:
        return self.lookup(ids, self.device)

    def begin(self) -> None:
        self.streamer.open()

    def end(self) -> None:
        self.streamer.close()
        self.h2d_bytes = self.streamer.h2d_bytes

    def resident_bytes(self) -> int:
        kept = tensor_ids(self.layers) | self.emb_ids
        return tensor_bytes(
            t for t in module_tensors(self.model).values() if id(t) not in kept
        )

    def pass_bytes(self) -> int:
        return self.resident_bytes() + self.streamer.slot_bytes()

    def release(self) -> None:
        self.streamer.remove()
        super().release()


PLACEMENTS = {
    "resident": TextPlacement,
    "cpu": HostPlacement,
    "offload": OffloadPlacement,
    "ring": RingPlacement,
}
QUANTIZE = ("int8", "fp8", *COMPUTE)


def host_lookup(
    embedding: nn.Module | None, dtypes: Mapping[int, torch.dtype] | None = None
) -> Callable | None:
    """`(ids, device) -> embeddings on device` of a table kept where it lives: the
    stored int8 rows of a QuantEmbedding, else the dense rows in the table's compute
    dtype (`device_dtype` of `dtypes`)."""
    if embedding is None:
        return None
    if isinstance(embedding, QuantEmbedding):
        return embedding.lookup
    return partial(
        dense_lookup, embedding, dtype=device_dtype(embedding.weight, dtypes)
    )


def decode_w4a8(
    text_model: nn.Module, work: torch.device, home: torch.device | None = None
) -> int:
    """Decode every W4A8 base of a text model's QuantLinears into its int8 QuantWeight
    (`W4A8Weight.to_int8`, on `work`, the result on `home`, default where the base
    was); returns how many."""
    count = 0
    for m in text_model.modules():
        if isinstance(m, QuantLinear) and isinstance(m.qweight, W4A8Weight):
            if m.qweight.decodes_to_int8:
                to = m.qweight.q.device if home is None else home
                m.qweight = m.qweight.to(work).to_int8().to(to)
                count += 1
    return count


def w4a8_extra_bytes(text_model: nn.Module) -> int:
    """Bytes the int8 decode of every W4A8 base adds over its packed form."""
    return sum(
        m.qweight.int8_nbytes() - m.qweight.nbytes()
        for m in text_model.modules()
        if isinstance(m, QuantLinear) and isinstance(m.qweight, W4A8Weight)
    )


@torch.no_grad()
def apply_to_text_encoder(
    text_model: nn.Module,
    plan: TextEncoderPlan,
    device: torch.device | str,
    compute_dtype: torch.dtype,
    choose: Callable[[nn.Module, torch.device], str] | None = None,
    source: SourceWeights | None = None,
    choose_w4a8: Callable[[nn.Module, torch.device], str] | None = None,
) -> tuple[TextPlacement, dict, TextEncoderPlan]:
    """Quantize a frozen text model's linears under `plan.precision` (per channel or
    per `quant_group`; from the checkpoint's stored values in `source` where they
    apply), settle the W4A8 decode (`plan.w4a8`; `auto` by `choose_w4a8(model,
    device)`: `int8` decodes once, `packed` per call), resolve `memory="auto"` with
    `choose(model, device)` on the quantized sizes, place it (resident / offload /
    ring / cpu) and bind the quantized GEMMs to the compute device. Returns
    (placement, quantization report, plan)."""
    report = {"stored": plan.stored} if plan.stored else {}
    if plan.precision in QUANTIZE and not plan.stored:
        report = quantize_text_linears(
            text_model,
            plan.precision,
            plan.quant_group,
            compute_dtype,
            recipe=plan.fp8_recipe,
            work_device=work_device(device),
            source=source,
        )
    if plan.precision == "w4a8":
        decode = plan.w4a8
        if decode == "auto":
            decode = choose_w4a8(text_model, torch.device(device))
        if decode == "int8":
            work = work_device(device)
            report["w4a8_decoded"] = decode_w4a8(text_model, work)
        plan = replace(plan, w4a8=decode)
    if plan.memory == "auto":
        plan = replace(plan, memory=choose(text_model, torch.device(device)))
    placement = place_text(text_model, plan, device, compute_dtype)
    return placement, report, plan


def place_text(
    text_model: nn.Module,
    plan: TextEncoderPlan,
    device: torch.device | str,
    compute_dtype: torch.dtype,
    host: str = "copy",
    dtypes: Mapping[int, torch.dtype] | None = None,
    deferred: Mapping[int, Deferred] | None = None,
) -> TextPlacement:
    """Place a (quantized) text model under `plan.memory` (resolved) with the host form
    `host` of its host-held weights (`deferred`: the tensors built on the device from
    a stored base), and bind its quantized GEMMs to the compute device."""
    layers = list(getattr(text_model, "layers", []))
    embedding = getattr(text_model, "embed_tokens", None)
    kw = {"slots": plan.stream_resident} if plan.memory == "ring" else {}
    placement = PLACEMENTS[plan.memory](
        text_model,
        layers,
        embedding,
        device,
        plan.pinned,
        host=host,
        dtypes=dtypes,
        deferred=deferred,
        **kw,
    )
    rebind_quant(text_model, placement.device, compute_dtype, autograd=False)
    return placement
