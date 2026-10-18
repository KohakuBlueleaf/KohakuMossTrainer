"""Records and tables in the log: a plan renders as a header line plus an aligned
key / value block (never a raw repr), defaults hidden at INFO and shown at DEBUG,
nested mappings indented, tables column-aligned, plain without a colour console and
per-field coloured with one."""

import io
import logging
from dataclasses import dataclass, field

import pytest

from kohakumoss.types.plans import MemoryPlan
from kohakumoss_common import log as klog
from tests._util.logs import collecting
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
RESET = "\033[0m"


@pytest.fixture(autouse=True)
def _plain_env(monkeypatch):
    for name in ("FORCE_COLOR", "NO_COLOR", klog.LEVEL_ENV):
        monkeypatch.delenv(name, raising=False)


def _block(text: str, title: str) -> None:
    """A header line, then indented `key  value` lines whose values share one column."""
    lines = text.splitlines()
    assert lines[0].startswith(title), lines
    assert "(" not in lines[0].partition(":")[2].split(" on ")[0], lines[0]
    body = [ln for ln in lines[1:] if not ln.strip().startswith("(")]
    assert body and all(ln.startswith("  ") for ln in body), lines
    starts = {len(ln) - len(ln[2:].split("  ", 1)[1].lstrip()) for ln in body}
    assert len(starts) == 1, lines


def test_memory_plan_renders_as_a_block():
    plan = MemoryPlan(profile="balanced", checkpoint="none", stream="blocks")
    with collecting(logging.INFO) as logs:
        klog.log_object(klog.get_logger("kohakumoss.t"), "DiT memory plan", plan)
    text = logs.messages[-1]
    assert text.splitlines()[0] == "DiT memory plan: balanced", text
    _block(text, "DiT memory plan")
    assert "MemoryPlan(" not in text
    assert "  checkpoint  none" in text and "  stream      blocks" in text
    n = len(MemoryPlan.__dataclass_fields__)
    assert text.splitlines()[-1] == f"  ({n - 2} defaults hidden)", text
    old = f"DiT memory profile balanced on cuda: {plan}: done"
    must_fail(lambda: _block(old, "DiT memory plan"), "the old repr line")
    must_fail(lambda: _absent(old, "MemoryPlan("), "a raw repr")


def _absent(text: str, needle: str) -> None:
    assert needle not in text, text


def test_defaults_hidden_at_info_shown_at_debug():
    plan = MemoryPlan(profile="lowvram", act_offload=True)
    log = klog.get_logger("kohakumoss.t2")
    with collecting(logging.INFO) as logs:
        klog.log_object(log, "plan", plan)
    info = logs.messages[-1].splitlines()
    n = len(MemoryPlan.__dataclass_fields__)
    assert info[-1] == f"  ({n - 2} defaults hidden)", info
    assert len(info) == 1 + 2 + 1, info
    with collecting(logging.DEBUG) as logs:
        klog.log_object(log, "plan", plan)
    debug = logs.messages[-1].splitlines()
    assert len(debug) == 1 + n and "defaults hidden" not in debug[-1], debug
    must_fail(lambda: _equal(len(info), len(debug)), "defaults shown at INFO")


def _equal(a, b) -> None:
    assert a == b


@dataclass
class Inner:
    rows: int = 4


@dataclass
class Outer:
    name: str = "x"
    inner: Inner = field(default_factory=Inner)
    kernels: dict = field(default_factory=dict)


def test_nested_records_indent():
    lines = klog.render(Outer(kernels={"attention": "triton", "rms_norm": "eager"}))
    assert lines == [
        "  name     x",
        "  inner",
        "    rows  4",
        "  kernels",
        "    attention  triton",
        "    rms_norm   eager",
    ], lines
    flat = [ln.strip() for ln in lines]
    must_fail(lambda: _equal(flat, lines), "nesting not indented")


def test_table_alignment():
    rows = [
        ("rms_norm_mod", 0.53, 12),
        ("qk_norm_rope", 12.5, 3),
        ("situ_glu", 1.0, 120),
    ]
    out = klog.table(("op", "secs", "calls"), rows)
    assert out == [
        "  op           | secs | calls",
        "  -------------+------+------",
        "  rms_norm_mod | 0.53 |    12",
        "  qk_norm_rope | 12.5 |     3",
        "  situ_glu     |    1 |   120",
    ], out
    bars = {ln.index("|") for ln in out if "|" in ln}
    assert len(bars) == 1
    ragged = [f"  {a} | {b} | {c}" for a, b, c in rows]
    must_fail(lambda: _aligned(ragged), "unaligned columns")
    _aligned(out[2:])


def test_table_list_cell_spans_lines():
    rows = [
        ("linear_gres", 30, ["BM=128,BN=128", "split"], [10, 9]),
        ("rms", 2, "a", 2),
    ]
    out = klog.table(("kernel", "measured", "chosen", "x"), rows)
    assert out == [
        "  kernel      | measured | chosen        | x",
        "  ------------+----------+---------------+---",
        "  linear_gres |       30 | BM=128,BN=128 | 10",
        "              |          | split         |  9",
        "  rms         |        2 | a             |  2",
    ], out
    joined = [("linear_gres", 30, "BM=128,BN=128 x10, split x9")]
    flat = klog.table(("kernel", "measured", "chosen"), joined)
    must_fail(lambda: _assert_lines(flat, 4), "one line per config")
    _assert_lines(out, 5)


def _assert_lines(out: list[str], n: int) -> None:
    assert len(out) == n, out


def _aligned(lines: list[str]) -> None:
    assert len({ln.index("|") for ln in lines}) == 1, lines


def test_plain_and_coloured_rendering(monkeypatch):
    plan = MemoryPlan(profile="balanced", act_offload=True, stream_resident=8)
    lines = klog.render(plan, defaults=False)
    assert all("\033[" not in ln for ln in lines), lines
    monkeypatch.setenv("FORCE_COLOR", "1")
    klog.setup("INFO", stream=io.StringIO())
    coloured = klog.render(plan, defaults=False)
    joined = "\n".join(coloured)
    assert f"\033[2;36mact_offload{RESET}" in joined
    assert f"\033[32mTrue{RESET}" in joined and f"\033[93m8{RESET}" in joined
    plain = [klog.SGR.sub("", ln) for ln in coloured]
    assert plain == lines
    must_fail(lambda: _absent(joined, "\033["), "a colour console rendered plain")
