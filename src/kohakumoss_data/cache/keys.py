"""Cache keys over every input that changes the cached tensor."""

from kohakumoss_data.seeds import text_hash
from kohakumoss_data.types import CatalogRow, PlanItem


def latent_key(
    row: CatalogRow,
    item: PlanItem,
    encoder_id: str,
    dtype: str,
    background: str,
    resample: str = "lanczos",
    draw: str = "mode",
) -> str:
    """Image content x encoder x storage dtype x target x crop x flip x background
    (alpha images only) x resample filter x VAE draw (`mode` or `sample`)."""
    content = row.extra.get("hash") or f"{row.locator}:{row.w}x{row.h}"
    return text_hash(
        "latent",
        content,
        encoder_id,
        dtype,
        item.target_hw,
        item.crop_ltrb,
        int(item.flip),
        background if row.has_alpha else "",
        resample,
        draw,
    )


def text_key(caption: str, encoder_id: str, dtype: str) -> str:
    """Rendered caption x text encoder x storage dtype."""
    return text_hash("text", encoder_id, dtype, caption)
