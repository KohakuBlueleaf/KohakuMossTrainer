"""Over-long fits: `(item, image_tokens, text_tokens, max_tokens, keep_seed)` -> the
fitted item, or None (dropped). A fitted sample keeps at least one text and one image
token."""

from dataclasses import replace


def drop(item, img: int, text: int, max_tokens: int, keep_seed: int):
    """Drop the sample."""
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

__all__ = ["OVERLONG", "drop", "drop_tokens", "truncate_text"]
