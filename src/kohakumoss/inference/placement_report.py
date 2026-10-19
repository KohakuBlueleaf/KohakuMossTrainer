"""Load-time placement line of a pipeline's DiT (where the weights live and compute,
with sizes), plus the summary table rows and the estimated GPU peak of a reference
generate; the encoder lines are kohakumoss.offload.encoders.report. See
docs/guides/logging.md."""

from kohakumoss.inference import peaks
from kohakumoss.logging.log import hl
from kohakumoss.offload.encoders.report import gib_text, split_bytes
from kohakumoss.offload.weights.layers import module_bytes
from kohakumoss.quant.densify import stored_precision_of
from kohakumoss_common.units import GIB
from kohakumoss_data.resolution.geometry import LATENT_FACTOR
from kohakumoss_kernels.dtypes import NAME_OF

TEXT_TOKENS = 512


def forward_tokens(
    sizes, batch: int, cfg: float, cfg_mode: str, text: int = TEXT_TOKENS
) -> int:
    """Tokens of the largest packed DiT forward of a generate call: `batch` samples of
    the largest of `sizes` ((H, W) pixels) with `text` text tokens each, doubled by
    packed CFG (cfg != 1)."""
    image = max((h // LATENT_FACTOR) * (w // LATENT_FACTOR) for h, w in sizes)
    factor = 2 if cfg_mode == "packed" and cfg != 1.0 else 1
    return factor * batch * (image + text)


REFERENCE_TOKENS = forward_tokens([(1024, 1024)], 1, 3.0, "packed")


def _fold(backbone) -> str:
    scales = getattr(backbone, "fold_scales", None)
    if getattr(backbone, "fp16_fix", "none") != "weights" or not scales:
        return ""
    c = float(scales[0])
    inv = 1.0 / c if c else 0.0
    text = f"1/{round(inv)}" if inv and abs(inv - round(inv)) < 1e-6 else f"{c:.4g}"
    return f", fp16 fold c={text} from metadata"


def dit_line(backbone, path, device, plan) -> tuple[str, dict]:
    """The transformer: resident on the device, or streamed from the host."""
    on, host = split_bytes(backbone, device)
    dtype = NAME_OF.get(next(backbone.parameters()).dtype, "mixed")
    precision = plan.base_precision
    if precision == "auto":
        precision = stored_precision_of(backbone)
    base = "" if precision == "compute" else f", base {precision}"
    head = f"transformer: {hl(path, 'path')} -> "
    fields = {"gpu_gib": round(on / GIB, 2), "host_gib": round(host / GIB, 2)}
    if host == 0:
        where = (
            f"resident on {device} ({hl(gib_text(on))}, {dtype}{base}{_fold(backbone)})"
        )
        return head + where, fields
    blocks = list(backbone.layers)
    resident = sum(split_bytes(b)[1] == 0 for b in blocks)
    where = (
        f"streamed from host, {hl(resident)} of {len(blocks)} blocks resident "
        f"({hl(gib_text(on))} on {device}, {hl(gib_text(host))} host; {dtype}{base}"
        f"{_fold(backbone)})"
    )
    return head + where, fields


def summary_rows(pipe) -> tuple[str, list[tuple]]:
    """The placement table's title (the estimated GPU peak of one 1024x1024 image with
    CFG) and its rows: (component, placement, GPU GiB, host GiB)."""
    te, vae = pipe.place.text_encoder, pipe.place.vae
    est = peaks.estimated(
        pipe.backbone, pipe.memory_plan, REFERENCE_TOKENS, pipe.place, 1024 * 1024
    )
    on, host = split_bytes(pipe.backbone, pipe.device)
    rows = [("DiT", "resident" if host == 0 else "streamed", _g(on), _g(host))]
    if te is not None and hasattr(te, "placement"):
        _, te_host = split_bytes(te.model)
        name = te.placement.name
        rows.append(("text encoder", name, _g(te.placement.pass_bytes()), _g(te_host)))
    if vae is not None:
        size = module_bytes(vae.vae)
        name = vae.placement.name + (" tiled" if vae.plan.tile else "")
        rows.append(
            ("VAE", name, _g(size), 0.0)
            if pipe.place.vae_on_device
            else ("VAE", name, 0.0, _g(size))
        )
    peak = f"{est['peak']:.2f} GiB"
    title = f"placement: estimated GPU peak {hl(peak)} (one 1024x1024 image, CFG)"
    return title, rows


def _g(n: int) -> float:
    return round(n / GIB, 2)


SUMMARY_HEADERS = ("component", "placement", "GPU GiB", "host GiB")
