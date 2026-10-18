"""Text-encoder prewarm bucket planner, model-free: the planned packs cover every
delta-rule and token bucket a plan can produce, and the data's longest caption caps
the rows at its length bucket."""

import random
from types import SimpleNamespace

import pytest

from kohakumoss.encoders.text import prewarm
from kohakumoss.encoders.text.prewarm import longest_row, pack_lengths
from kohakumoss.encoders.text.tokens import plan_groups
from kohakumoss_kernels.tuning.buckets import delta_buckets
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _live_buckets(batch: int, max_tokens: int, max_len: int, seed: int) -> set:
    rng = random.Random(seed)
    out = set()
    for _ in range(400):
        lengths = [rng.randint(1, max_len) for _ in range(rng.randint(1, 3 * batch))]
        for g in plan_groups(lengths, batch, max_tokens, False):
            rows = [lengths[i] for i in g]
            out.add(delta_buckets(sum(rows), len(rows), max(rows)))
    return out


def check_planner_covers(batch: int, max_tokens: int, max_len: int) -> None:
    packs = pack_lengths(batch, max_tokens, max_len)
    covered = {delta_buckets(sum(p), len(p), max(p)) for p in packs}
    missing = _live_buckets(batch, max_tokens, max_len, 1) - covered
    assert not missing, sorted(missing)
    for p in packs:
        assert len(p) <= batch and sum(p) <= max_tokens and max(p) <= max_len, p


@pytest.mark.parametrize(
    "batch, max_tokens, max_len", [(8, 512, 152), (32, 16384, 527), (1, 4096, 300)]
)
def test_bucket_planner_covers_live_packs(batch, max_tokens, max_len):
    check_planner_covers(batch, max_tokens, max_len)


TOKENIZER = SimpleNamespace(token_ids=lambda prompts: [[0] * 20], max_length=8192)


def check_rows_within(max_text_tokens: int, data_tokens: int, cap: int) -> None:
    """Every planned pack row is at most `cap` tokens (32-row packs, 16384 tokens)."""
    rows = longest_row(TOKENIZER, max_text_tokens, data_tokens)
    packs = pack_lengths(32, 16384, rows)
    assert max(max(p) for p in packs) <= cap, rows


def test_data_caps_the_longest_row():
    assert longest_row(TOKENIZER, 4096) == 4096 + 20
    assert longest_row(TOKENIZER, 4096, 300) == 512
    assert longest_row(TOKENIZER, 200, 300) == 220
    check_rows_within(4096, 300, 512)
    must_fail(lambda: check_rows_within(4096, 0, 512), "rows up to MAX_TEXT_TOKENS")


def test_negative_planner_without_mixed_rows(monkeypatch):
    """Packs of equal rows only (mean == longest) leave short-mean buckets out."""
    monkeypatch.setattr(
        prewarm, "_with_mean", lambda longest, count, mean: (longest,) * count
    )
    must_fail(lambda: check_planner_covers(32, 16384, 527), "equal rows only")
