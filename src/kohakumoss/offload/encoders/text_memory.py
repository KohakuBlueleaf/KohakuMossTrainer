"""Text-encoder device-memory model: weights per placement and the activations of one
pass, and the `auto` placement choice. See docs/internals/offload.md."""

from dataclasses import dataclass

from torch import nn

from kohakumoss.offload.weights.layers import module_bytes, module_tensors, tensor_bytes

ORDER = ("resident", "ring")


@dataclass(frozen=True)
class TextSizes:
    """Byte sizes of a loaded text model and its shape constants."""

    total: int
    embedding: int
    layers: int
    ring: int
    hidden: int
    intermediate: int
    value_heads: int
    head_k: int
    head_v: int
    elem: int


def text_sizes(model: nn.Module, slots: int = 2, extra_layers: int = 0) -> TextSizes:
    """Sizes of a (possibly quantized) Qwen3.5 text model; `ring` is the layer ring of
    `slots` slots per layer signature; `extra_layers` bytes are added to the decoder
    layers (spread evenly over the ring's signatures: a decode the model has not run
    yet)."""
    cfg = model.config
    layers = list(model.layers)
    sigs: dict[tuple, int] = {}
    total_layers = 0
    for layer in layers:
        ts = module_tensors(layer)
        sig = tuple((n, tuple(t.shape), t.dtype) for n, t in ts.items())
        sigs[sig] = tensor_bytes(ts.values())
        total_layers += sigs[sig]
    per_layer = extra_layers / max(len(layers), 1)
    return TextSizes(
        total=module_bytes(model) + extra_layers,
        embedding=module_bytes(model.embed_tokens),
        layers=total_layers + extra_layers,
        ring=int(max(slots, 2) * (sum(sigs.values()) + per_layer * len(sigs))),
        hidden=cfg.hidden_size,
        intermediate=cfg.intermediate_size,
        value_heads=cfg.linear_num_value_heads,
        head_k=cfg.linear_key_head_dim,
        head_v=cfg.linear_value_head_dim,
        elem=model.norm.weight.element_size(),
    )


def pass_activations(s: TextSizes, tokens_pass: int, tokens_call: int) -> int:
    """Peak activation bytes of one kernel-forward pass: every pack's hidden state and
    residual, plus one call's MLP / projection intermediates and delta-rule scratch
    (chunk start states fp32 per 64 tokens, UT transforms and V')."""
    hidden = 2 * tokens_pass * s.hidden * s.elem
    call = tokens_call * (3 * s.intermediate + 6 * s.hidden) * s.elem
    per_tok_state = s.head_k * s.head_v * 4 / 64 + 64 * 4 + s.head_v * 4
    scratch = int(tokens_call * s.value_heads * per_tok_state)
    return hidden + call + scratch


def text_footprint(memory: str, s: TextSizes, tokens_pass: int, tokens_call: int) -> dict[str, int]:  # fmt: skip
    """Device bytes held between passes (`resident`) and at the peak of a pass (`pass`)."""
    act = pass_activations(s, tokens_pass, tokens_call)
    match memory:
        case "resident":
            return {"resident": s.total, "pass": s.total + act}
        case "offload":
            return {"resident": 0, "pass": s.total - s.embedding + act}
        case "ring":
            rest = s.total - s.embedding - s.layers
            return {"resident": rest, "pass": rest + s.ring + act}
        case _:
            return {"resident": 0, "pass": 0}


def choose_text_memory(
    s: TextSizes, budget: int, step_peak: int, tokens_pass: int, tokens_call: int
) -> tuple[str, list[dict]]:
    """The first placement in ORDER that fits: resident needs its weights next to the
    DiT step peak, ring its pass peak. Returns (choice, per-placement rows)."""
    rows = []
    for memory in ORDER:
        fp = text_footprint(memory, s, tokens_pass, tokens_call)
        need = max(step_peak + fp["resident"], fp["pass"])
        rows.append({"memory": memory, **fp, "need": need, "fits": need <= budget})
        if need <= budget:
            return memory, rows
    return "ring", rows
