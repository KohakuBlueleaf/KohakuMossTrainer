"""Over-long policies: what happens to a single sample longer than `max_tokens`.

Each policy maps `(item, image_tokens, text_tokens, max_tokens, keep_seed)` to the fitted
item or None (dropped). A sample keeps at least one text token and one image token;
when that cannot fit, it is dropped.
"""

from dataclasses import replace

from kohakumoss_data.resolution.item import MossPlanItem


def drop(item, img: int, text: int, max_tokens: int, keep_seed: int):
    return None


def truncate_text(item, img: int, text: int, max_tokens: int, keep_seed: int):
    """Keep the first `max_tokens - img` text tokens."""
    limit = max_tokens - img
    return None if limit < 1 else replace(item, text_limit=limit)


def drop_tokens(item, img: int, text: int, max_tokens: int, keep_seed: int):
    """Keep `max_tokens - text` image tokens, chosen by a seeded draw."""
    keep = max_tokens - text
    return None if keep < 1 else replace(item, img_keep=keep, keep_seed=keep_seed)


OVERLONG = {"drop": drop, "truncate_text": truncate_text, "drop_tokens": drop_tokens}

__all__ = ["OVERLONG", "MossPlanItem", "drop", "drop_tokens", "truncate_text"]
