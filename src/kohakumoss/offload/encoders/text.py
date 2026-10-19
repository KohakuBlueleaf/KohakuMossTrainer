"""Where a frozen text encoder's weights live between and during its passes.

`resident`: everything on the device. `cpu`: everything on the host, run there.
`offload`: host (pinned) copies; every pass copies the whole model in and drops the
device copies after it. `ring`: decoder layers through a `LayerStreamer` ring, the
small rest resident. Under offload / ring the token embedding table stays on the
host and only the looked-up rows cross the bus. `begin()` / `end()` bracket one pass;
`end()` releases every device weight buffer the pass used. See
docs/internals/offload.md.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss.offload.accel import streamed
from kohakumoss.offload.encoders.placement import HostCopies
from kohakumoss.offload.weights.layers import (
    LayerStreamer,
    host_copy,
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
    ):
        self.model = model
        self.layers = list(layers)
        self.embedding = embedding
        self.emb_ids = frozenset() if embedding is None else tensor_ids([embedding])
        self.lookup = host_lookup(embedding)
        self.device = torch.device(device)
        self.pinned = pinned and streamed(self.device)
        self.h2d_bytes = 0
        self.enter: list[Callable[[], None]] = [nothing] * len(self.layers)
        self.leave: list[Callable[[], None]] = [nothing] * len(self.layers)
        self.place()

    def place(self) -> None:
        self.model.to(self.device)

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


class HostPlacement(TextPlacement):
    """Everything on the host; the encoder runs there."""

    name = "cpu"

    def __init__(self, model, layers, embedding, device="cpu", pinned=False):
        super().__init__(model, layers, embedding, "cpu", False)


class OffloadPlacement(TextPlacement):
    """Host copies; each pass moves the whole model (bar the embedding) in."""

    name = "offload"

    def place(self) -> None:
        tensors = list(module_tensors(self.model).values())
        for t in tensors:
            if id(t) in self.emb_ids:
                t.data = host_copy(t, False)
        moved = [t for t in tensors if id(t) not in self.emb_ids]
        self.copies = HostCopies(moved, self.device, self.pinned)

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


class RingPlacement(TextPlacement):
    """Decoder layers streamed through `slots` ring slots per layer signature; the
    final norm and rotary tables resident; the embedding table on the host."""

    name = "ring"

    def __init__(self, model, layers, embedding, device, pinned=True, slots=2):
        self.slots = max(int(slots), 2)
        super().__init__(model, layers, embedding, device, pinned)

    def place(self) -> None:
        inside = tensor_ids(self.layers)
        for t in module_tensors(self.model).values():
            if id(t) in self.emb_ids:
                t.data = host_copy(t, False)
            elif id(t) not in inside:
                t.data = t.data.to(self.device)
        self.streamer = LayerStreamer(
            self.layers, self.device, slots=self.slots, pinned=self.pinned
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


def host_lookup(embedding: nn.Module | None) -> Callable | None:
    """`(ids, device) -> embeddings on device` of a table kept where it lives: the
    stored int8 rows of a QuantEmbedding, else the dense rows."""
    if embedding is None:
        return None
    if isinstance(embedding, QuantEmbedding):
        return embedding.lookup
    return partial(dense_lookup, embedding)


def decode_w4a8(text_model: nn.Module, work: torch.device) -> int:
    """Decode every W4A8 base of a text model's QuantLinears into its int8 QuantWeight
    (`W4A8Weight.to_int8`, on `work`, the result back where the base was); returns
    how many."""
    count = 0
    for m in text_model.modules():
        if isinstance(m, QuantLinear) and isinstance(m.qweight, W4A8Weight):
            if m.qweight.decodes_to_int8:
                home = m.qweight.q.device
                m.qweight = m.qweight.to(work).to_int8().to(home)
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
    layers = list(getattr(text_model, "layers", []))
    embedding = getattr(text_model, "embed_tokens", None)
    kw = {"slots": plan.stream_resident} if plan.memory == "ring" else {}
    placement = PLACEMENTS[plan.memory](
        text_model, layers, embedding, device, plan.pinned, **kw
    )
    rebind_quant(text_model, placement.device, compute_dtype, autograd=False)
    return placement, report, plan
