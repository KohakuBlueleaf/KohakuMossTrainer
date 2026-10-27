"""The encoder process: the text encoder and the VAE on one encoder card, in a process
of their own (own CUDA context, kernel registry and tuner). Every encoder process
takes chunks from the feed's one request queue, encodes each chunk in one text-encoder
call and one VAE call, and answers on the feed's result queue. See
docs/internals/encode-feed.md."""

import os
import queue
import time
import traceback

import torch
import torch.multiprocessing as mp

from kohakumoss.encoders.text import QwenTextEncoder
from kohakumoss.encoders.vae import FluxVae, resolve_vae_dtype
from kohakumoss.logging.log import get_logger, hl, setup
from kohakumoss.offload.accel import configure_allocator
from kohakumoss.offload.encoders.vae import place_vae
from kohakumoss.training.build.kernels import TextPrewarm
from kohakumoss_kernels import compile_setup
from kohakumoss_kernels.tuning.tuner import set_mode as set_tune_mode
from kohakumoss_kernels.tuning.window import tuning_window

log = get_logger(__name__)
CONTEXT = "spawn"
SHARING = "file_system"
READY = "ready"
FAILED = "failed"
POLL_S = 5.0
PROCESS_KNOBS = (
    "TUNE",
    "MAX_TEXT_TOKENS",
    "VAE_DTYPE",
    "LATENT_DRAW",
    "VAE_ENCODE",
    "TARGET_AREA",
)


def _build(spec: dict):
    """(text encoder | None, VAE | None) of `spec` on its card, the text encoder
    prewarmed before its first pass."""
    k = spec["knobs"]
    device = torch.device(spec["device"])
    te = vae = None
    if spec["text_path"]:
        te = QwenTextEncoder.from_pretrained(
            spec["text_path"],
            device,
            spec["text_plan"],
            output_device=torch.device("cpu"),
            run_device=device,
        )
        if (k.get("TUNE") or "table") != "off":
            te.before_first(TextPrewarm(te, int(k["MAX_TEXT_TOKENS"])))
    if spec["vae_path"]:
        dtype = resolve_vae_dtype(k.get("VAE_DTYPE") or "auto", device)
        vae = place_vae(
            FluxVae.from_pretrained(spec["vae_path"], "cpu", dtype, k["LATENT_DRAW"]),
            {"memory": "resident", **dict(k.get("VAE_ENCODE") or {})},
            device,
            0,
            int(k["TARGET_AREA"]),
        )
    return te, vae


def pack(parts: list) -> tuple:
    """Per-prompt [L_i, D] host tensors as one [sum L, D] tensor and the lengths."""
    if not parts:
        return None, []
    return torch.cat(parts, 0).contiguous(), [p.shape[0] for p in parts]


def unpack(packed: tuple) -> list:
    """`pack`'s inverse."""
    flat, lengths = packed
    return [] if flat is None else list(flat.split(lengths, 0))


def encode_texts(te, items: list[dict]) -> list[tuple]:
    """Every item's prompts (cut to their limits) in one encoder call; one packed
    (flat, lengths) per item."""
    prompts = [p for it in items for p in it["prompts"]]
    if not prompts:
        return [(None, [])] * len(items)
    limits = [n for it in items for n in it["limits"]]
    flat = [t.cpu() for t in te.encode(prompts, limits)]
    out, at = [], 0
    for it in items:
        n = len(it["prompts"])
        out.append(pack(flat[at : at + n]))
        at += n
    return out


def image_order(items: list[dict]) -> list[tuple[int, int]]:
    """(item, image) of every image of `items`, sorted by latent size so equal sizes
    run together through the VAE."""
    refs = [
        (i, j)
        for i, it in enumerate(items)
        if it["pixels"] is not None
        for j in range(len(it["pixels"]))
    ]
    return sorted(refs, key=lambda r: tuple(items[r[0]]["hw"][r[1]]))


def encode_images(vae, items: list[dict]) -> list:
    """Every item's images in one VAE call (size-sorted); the packed full-grid latents
    per item in its own image order (None for an item without images)."""
    order = image_order(items)
    if not order:
        return [None] * len(items)
    sizes = [tuple(items[i]["hw"][j]) for i, j in order]
    z = vae.encode_images(
        [items[i]["pixels"][j] for i, j in order], torch.tensor(sizes)
    )
    parts = z.cpu().split([h * w for h, w in sizes], 0)
    each: dict = {ref: part for ref, part in zip(order, parts)}
    out = []
    for i, it in enumerate(items):
        if it["pixels"] is None:
            out.append(None)
            continue
        out.append(torch.cat([each[(i, j)] for j in range(len(it["pixels"]))], 0))
    return out


def encode_chunk(te, vae, items: list[dict]) -> dict:
    """The result payload of one chunk: per-item text and latents, and the encode
    seconds and counts of the chunk."""
    start = time.perf_counter()
    text = encode_texts(te, items) if te is not None else [(None, [])] * len(items)
    te_s = time.perf_counter() - start
    start = time.perf_counter()
    lat = encode_images(vae, items) if vae is not None else [None] * len(items)
    vae_s = time.perf_counter() - start
    return {
        "text": text,
        "latents": lat,
        "te_s": te_s,
        "vae_s": vae_s,
        "tokens": sum(sum(lengths) for _, lengths in text),
        "images": sum(len(it["pixels"]) for it in items if it["pixels"] is not None),
    }


def _ready(te, spec: dict) -> dict:
    samples = None
    if spec["sample_texts"] and te is not None:
        with tuning_window():
            samples = pack([t.cpu() for t in te.encode(spec["sample_texts"])])
    flops = 0 if te is None else te.flops_per_token
    return {"device": spec["device"], "samples": samples, "flops_per_token": flops}


def serve(spec: dict, requests, results) -> None:
    """The process body: build, answer `(READY, info)`, then one `(id, payload)` per
    request `(id, items)` until `None`; a failure answers `(FAILED, (device,
    traceback))` and ends the process; it also ends when its parent is gone."""
    try:
        setup()
        mp.set_sharing_strategy(SHARING)
        device = torch.device(spec["device"])
        configure_allocator(device)
        compile_setup.apply()
        set_tune_mode(spec["knobs"].get("TUNE") or "table")
        torch.cuda.set_device(device)
        te, vae = _build(spec)
        results.put((READY, _ready(te, spec)))
        log.info(f"encoder process on {hl(spec['device'])} ready")
        while True:
            try:
                item = requests.get(timeout=POLL_S)
            except queue.Empty:
                if os.getppid() != spec["parent"]:
                    return
                continue
            if item is None:
                return
            rid, items = item
            results.put((rid, encode_chunk(te, vae, items)))
    except BaseException:  # noqa: BLE001 - reported to the feed, which fails the run
        results.put((FAILED, (spec["device"], traceback.format_exc())))
