"""The encoder-placement rows (no cache; text / VAE placements per setup, right after
the cache rows) and their report: the placement `auto` picked for the same setup comes
from that run's log."""

import pytest

pytest.importorskip(
    "kohakumoss_bench", reason="not installed: uv pip install -e scripts/bench/matrix"
)

from kohakumoss_bench.plan import matrix
from kohakumoss_bench.plan.config import Card, Host, Options
from kohakumoss_bench.report import results
from kohakumoss_bench.report.build import placement_rows
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
PER_POOL = 5 + 4 + 4


def host() -> Host:
    return Host(
        name="h",
        root="/r",
        python="python",
        weights={"source": "/w/base", "stage": "/s/base"},
        datasets=[{"source": "kohya", "path": "/d"}],
        cache_dir="/c",
    )


def check_queue(queue: list, card: Card) -> None:
    kinds = [r.kind for r in queue]
    rows = [r for r in queue if r.kind == "placement"]
    assert len(rows) == PER_POOL, (card.name, len(rows))
    first = kinds.index("placement")
    assert kinds[first : first + PER_POOL] == ["placement"] * PER_POOL
    assert all(k == "cache" for k in kinds[1:first]), "placement rows before cache rows"
    assert len({r.name for r in rows}) == PER_POOL
    for r in rows:
        assert r.knobs["LATENT_CACHE"] == "off" and r.knobs["TEXT_CACHE"] == "off"
        assert r.knobs["TEXT_ENCODE"]["memory"] == r.tags["text_memory"], r.name
        vae = r.knobs.get("VAE_ENCODE", {}).get("memory", "profile")
        assert vae == r.tags["vae_memory"], r.name
    lokr = [r for r in rows if r.tags["profile"] == matrix.LOWVRAM_LOKR_PROFILE]
    assert all(r.knobs["TEXT_ENCODE"]["ahead"] == 8 for r in lokr)
    vaes = {r.tags["vae_memory"] for r in rows if r.tags["profile"] == "balanced"}
    assert vaes == {"profile", "resident", "offload"}
    if not card.bf16:
        assert {r.tags["precision"] for r in rows} == {"fp16"}


def test_placement_rows_follow_the_cache_rows():
    for card in (Card(0, "5090", 32.0, 120), Card(0, "3090", 24.0, 86)):
        check_queue(matrix.card_queue(host(), Options(), card, []), card)
    turing = Card(0, "2080ti", 22.0, 75)
    check_queue(matrix.card_queue(host(), Options(), turing, []), turing)


def test_negative_queue_without_placement_rows():
    card = Card(0, "5090", 32.0, 120)
    queue = [
        r
        for r in matrix.card_queue(host(), Options(), card, [])
        if r.kind != "placement"
    ]
    must_fail(lambda: check_queue(queue, card), "no placement rows")


def _records(tmp_path, log_lines: str) -> list[dict]:
    log = tmp_path / "log.txt"
    log.write_text(log_lines, encoding="utf-8")
    summary = results.log_summary(log)
    base = "placement-bf16-compute-lora32-balanced"
    auto = {"kind": "placement", "run": f"{base}-teauto", "summary": summary}
    auto["tags"] = {"text_memory": "auto", "vae_memory": "profile"}
    other = {"kind": "placement", "run": f"{base}-teoffload", "summary": {}}
    other["tags"] = {"text_memory": "offload", "vae_memory": "profile"}
    return [auto, other]


def check_auto_column(lines: list[str]) -> None:
    assert all("| stream / resident |" in line for line in lines[2:]), lines


def test_auto_column_from_the_auto_runs_log(tmp_path):
    text = "[x] VAE placement: resident\n[x] text encoder placement: stream\n"
    check_auto_column(placement_rows(_records(tmp_path, text)))
    must_fail(
        lambda: check_auto_column(placement_rows(_records(tmp_path, "no lines\n"))),
        "no placement lines in the log",
    )
