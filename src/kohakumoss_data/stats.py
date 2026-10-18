"""Catalog and plan statistics (inspect script, bench, docs tables)."""

import collections

import numpy as np

from kohakumoss_data.resolution.base import expected_items


def catalog_stats(catalog) -> dict:
    """Live-row count, effective epoch size, pixel/aspect quantiles, alpha share,
    groups, captioned share; from the catalog columns only."""
    live = catalog.live
    w, h = catalog.w[live], catalog.h[live]
    area = (w * h).astype(np.float32)
    aspect = np.divide(w, h, dtype=np.float32)
    q = (0.05, 0.25, 0.5, 0.75, 0.95)
    text = sum(catalog.chars[k] for k in ("tags_chars", "nl_chars", "json_chars"))
    return {
        "rows": int(live.sum()),
        "epoch_items": expected_items(catalog.weight),
        "megapixels_q": [round(float(v) / 1e6, 3) for v in np.quantile(area, q)],
        "aspect_q": [round(float(v), 3) for v in np.quantile(aspect, q)],
        "alpha_share": float(np.mean(catalog.has_alpha[live])),
        "groups": len(np.unique(catalog.group_id[live])),
        "captioned_share": float(np.mean(text[live] > 0)),
    }


def plan_stats(dataset, epoch: int = 0, budget: int | None = None) -> dict:
    """Batches of one epoch: samples and tokens per batch, budget fill, crop loss,
    upscaled share, most common target sizes, over-long fits and drops."""
    planner = dataset.planner
    batches = planner.epoch(epoch)
    samples, img_tokens, txt_tokens, crop_loss, upscaled = [], [], [], [], 0
    sizes = collections.Counter()
    fitted = 0
    for batch in batches:
        samples.append(len(batch))
        split = [planner.sizes(it) for it in batch]
        img_tokens.append(sum(s[0] for s in split))
        txt_tokens.append(sum(s[1] for s in split))
        fitted += sum(bool(it.text_limit or it.img_keep) for it in batch)
        for it in batch:
            pixels = int(dataset.catalog.w[it.row_id] * dataset.catalog.h[it.row_id])
            left, top, right, bottom = it.crop_ltrb
            box = (right - left) * (bottom - top)
            crop_loss.append(1.0 - box / pixels)
            upscaled += it.target_hw[0] * it.target_hw[1] > box
            sizes[it.target_hw] += 1
    total = np.array(img_tokens) + np.array(txt_tokens)
    out = {
        "batches": len(batches),
        "samples_per_batch": float(np.mean(samples)),
        "image_tokens_per_batch": float(np.mean(img_tokens)),
        "text_tokens_per_batch": float(np.mean(txt_tokens)),
        "tokens_per_batch_q": [int(v) for v in np.quantile(total, (0.05, 0.5, 0.95))],
        "crop_loss_mean": float(np.mean(crop_loss)),
        "upscaled_share": upscaled / max(1, sum(samples)),
        "distinct_sizes": len(sizes),
        "overlong_fitted": fitted,
        "overlong_dropped": planner.drops.get(epoch, 0),
        "unbatched": planner.unbatched.get(epoch, 0),
        "top_sizes": [(f"{h}x{w}", n) for (h, w), n in sizes.most_common(8)],
    }
    if budget:
        out["budget_fill"] = float(np.mean(total / budget))
        out["padding_waste"] = 1.0 - out["budget_fill"]
    return out
