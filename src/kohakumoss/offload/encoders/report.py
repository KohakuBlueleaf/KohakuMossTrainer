"""Placement lines of the placed encoders (text encoder, VAE): where the weights live and
where they compute, with sizes (`(message, fields)` for `log.info(message, **fields)`).
Training and inference both log them. See docs/guides/logging.md."""

import torch
from torch import nn

from kohakumoss.logging.log import hl
from kohakumoss.offload.weights.layers import module_bytes
from kohakumoss_common.units import GIB


def split_bytes(module: nn.Module | None, device="cuda") -> tuple[int, int]:
    """(device bytes, host bytes) of a module's parameters and buffers; with a host
    `device` everything counts as on the device."""
    if module is None:
        return 0, 0
    if torch.device(device).type == "cpu":
        return module_bytes(module), 0
    on, host, seen = 0, 0, set()
    for t in [*module.parameters(), *module.buffers()]:
        if id(t) in seen:
            continue
        seen.add(id(t))
        if t.device.type == "cpu":
            host += t.nbytes
        else:
            on += t.nbytes
    return on, host


def gib_text(n: int | float) -> str:
    """`n` bytes as "x.xx GiB"."""
    return f"{n / GIB:.2f} GiB"


def te_line(te, path, device) -> tuple[str, dict]:
    """The text encoder: its placement, host RAM vs device bytes, precision."""
    placement = te.placement
    name = placement.name
    _, host = split_bytes(te.model)
    res, run = placement.resident_bytes(), placement.pass_bytes()
    dev = placement.device if name != "cpu" else torch.device("cpu")
    match placement.host:
        case "file":
            pinned = ", file pages"
        case _:
            pinned = ", pinned" if placement.pinned else ""
    precision = f"precision {te.plan.precision}, {te.plan.dtype}"
    match name:
        case "resident":
            where = f"resident on {dev} ({hl(gib_text(res))}; {precision})"
        case "cpu":
            where = (
                f"cpu (weights in host RAM, {hl(gib_text(host))}; runs on the host, "
                f"{hl(torch.get_num_threads())} threads; {precision})"
            )
        case "offload":
            where = (
                f"offload (weights in host RAM{pinned}, {hl(gib_text(host))}; whole model "
                f"moved to {dev} per pass, ~{hl(gib_text(run))}; {precision})"
            )
        case _:
            ring = getattr(placement, "slots", 0)
            where = (
                f"{name} (weights in host RAM{pinned}, {hl(gib_text(host))}; ring of "
                f"{hl(ring)} layers on {dev}, ~{hl(gib_text(run - res))}, "
                f"{hl(gib_text(res))} resident; {precision})"
            )
    if te.plan.free_after_use:
        where += "; freed after the first encode"
    fields = {
        "gpu_gib": round(res / GIB, 2),
        "gpu_pass_gib": round(run / GIB, 2),
        "host_gib": round(host / GIB, 2),
    }
    return f"text encoder: {hl(path, 'path')} -> {where}", fields


def vae_line(vae, path, placement: str, device) -> tuple[str, dict]:
    """The VAE under `placement`: resident on the device, parked on the host between
    passes (offload), its blocks streamed through a layer ring (ring), or run on the
    host (cpu)."""
    size = module_bytes(getattr(vae, "vae", vae))
    match placement:
        case "offload":
            where = (
                f"offload (host RAM, moved to {device} per pass, {hl(gib_text(size))})"
            )
        case "ring":
            where = (
                f"ring (host RAM, {hl(gib_text(size))}; encoder / decoder blocks through "
                f"a 2-layer ring on {device}, the rest moved per pass)"
            )
        case "cpu":
            where = (
                f"cpu (host RAM, {hl(gib_text(size))}; runs on the host, "
                f"{hl(torch.get_num_threads())} threads)"
            )
        case _:
            where = f"resident on {device} ({hl(gib_text(size))})"
    return f"VAE: {hl(path, 'path')} -> {where}", {"gib": round(size / GIB, 2)}


def encoder_lines(te, vae, te_path, vae_path, device) -> tuple:
    """(text-encoder line, VAE line) of a run or pipeline's placed encoders; None for
    an encoder that is absent or unplaced."""
    te_out = (
        te_line(te, te_path, device)
        if te is not None and hasattr(te, "placement")
        else None
    )
    if vae is None:
        return te_out, None
    name = getattr(getattr(vae, "placement", None), "name", "resident")
    return te_out, vae_line(vae, vae_path, name, getattr(vae, "device", device))
