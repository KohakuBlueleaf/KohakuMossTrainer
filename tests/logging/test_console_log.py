"""Console logging: the line format byte for byte, a colour per token (never one colour
over a line, the message itself uncoloured below WARNING), highlighted field values and
hl() spans, every span closed, a plain file handler, colour on a TTY unless NO_COLOR,
and always with FORCE_COLOR (a pipe too), the package root dropped from the module
column, structured fields with reserved names suffixed, the caller's location, an
idempotent setup, the rank filter, the level knob, stdlib loggers left their own class,
and progress bars that follow the level (model-free; the text-encoder bar and the
pipeline's phase messages run in tests/integration/test_infer_real.py)."""

import inspect
import io
import logging
import os
import subprocess
import sys
import time
from pathlib import PurePosixPath

import pytest

from kohakumoss.logging import log as km_log
from kohakumoss_common import log as klog
from tests._util.logs import Collect
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
RESET = "\033[0m"
LEVEL_CODES = [
    (logging.DEBUG, "\033[36m"),
    (logging.INFO, "\033[32m"),
    (logging.WARNING, "\033[33m"),
    (logging.ERROR, "\033[31m"),
    (logging.CRITICAL, "\033[1;37;41m"),
]


@pytest.fixture(autouse=True)
def _plain_env(monkeypatch):
    """No colour or rank switches leak in from the shell running the tests."""
    for name in ("FORCE_COLOR", "NO_COLOR", klog.LEVEL_ENV):
        monkeypatch.delenv(name, raising=False)


def record(level: int, name: str = "kohakumoss.x", **fields) -> logging.LogRecord:
    rec = logging.LogRecord(name, level, __file__, 1, "loaded", None, None)
    rec.created = 1_800_000_000.0
    rec.__dict__.update(fields)
    return rec


def clock(rec: logging.LogRecord) -> str:
    return time.strftime("%H:%M:%S", time.localtime(rec.created))


def _equal(a, b) -> None:
    assert a == b


def test_line_format():
    rec = record(logging.INFO, "kohakumoss.inference.pipeline", gib=9.6, secs=27.1)
    line = klog.Formatter(colour=False).format(rec)
    expected = f"[{clock(rec)}] [inference.pipeline] [INFO] loaded [gib=9.6, secs=27.1]"
    assert line == expected
    rec.rank = 3
    ranked = klog.Formatter(colour=False).format(rec)
    assert ranked == expected.replace("] [inf", "] [rank3] [inf", 1)
    short = klog.Formatter(colour=False).format(record(logging.WARNING))
    assert short == f"[{clock(rec)}] [x] [WARNING] loaded"
    deep = record(
        logging.INFO, "kohakumoss_kernels.backends.compile.prewarm_extra_long"
    )
    trimmed = klog.Formatter(colour=False).format(deep)
    assert "] [...pile.prewarm_extra_long] [INFO]" in trimmed, trimmed
    fits = record(logging.INFO, "kohakumoss_kernels.backends.compile.prewarm")
    assert "] [backends.compile.prewarm] [INFO]" in klog.Formatter(False).format(fits)
    outside = klog.Formatter(colour=False).format(record(logging.INFO, "scripts.x"))
    assert "] [scripts.x] [INFO]" in outside
    rooted = expected.replace("[inference", "[kohakumoss.inference")
    must_fail(lambda: _equal(line, rooted), "the package root left in the name")


def _closed(line: str) -> None:
    """Every SGR span is closed by a reset before the line ends."""
    open_ = False
    for code in klog.SGR.findall(line):
        open_ = code != RESET
    assert not open_, repr(line)


def _distinct(line: str) -> None:
    codes = set(klog.SGR.findall(line)) - {RESET}
    assert len(codes) >= 3, (codes, repr(line))


def _message(line: str, level: str) -> str:
    return line.partition(f"{level}{RESET}] ")[2]


def _plain(text: str) -> None:
    assert "\033[" not in text, repr(text)


@pytest.mark.parametrize("level, seq", LEVEL_CODES)
def test_each_token_has_its_colour(level, seq):
    name = logging.getLevelName(level)
    line = klog.Formatter(colour=True).format(record(level))
    stamp = clock(record(level))
    head = f"[\033[90m{stamp}{RESET}] [\033[35mx{RESET}] [{seq}{name}{RESET}] "
    assert line.startswith(head), repr(line)
    _distinct(line)
    _closed(line)
    message = _message(line, name)
    if level <= logging.INFO:
        assert message == "loaded"
        _uncoloured_message(line, name)
    else:
        assert message == f"{klog.TINTS[name]}loaded{RESET}"
    plain = klog.Formatter(colour=False).format(record(level))
    whole = f"{seq}{plain}{RESET}"
    must_fail(lambda: _distinct(whole), "one colour over the line")
    must_fail(lambda: _uncoloured_message(whole, name), "a coloured message")


def _uncoloured_message(line: str, level: str) -> None:
    """The message segment exists and carries no colour code."""
    message = _message(line, level)
    assert message, repr(line)
    _plain(message)


def test_field_values_highlighted():
    rec = record(
        logging.INFO,
        gib=9.6,
        path=PurePosixPath("ckpt/dit.safetensors"),
        plan="cpu",
        tokens="512 tokens",
    )
    line = klog.Formatter(colour=True).format(rec)
    key = "\033[2;36m"
    assert f"{key}gib{RESET}=\033[93m9.6{RESET}" in line, repr(line)
    assert f"{key}path{RESET}=\033[4;34mckpt/dit.safetensors{RESET}" in line
    assert f"{key}plan{RESET}=\033[1mcpu{RESET}" in line
    assert f"{key}tokens{RESET}=\033[93m512 tokens{RESET}" in line
    _plain(_message(line, "INFO").partition(" [")[0])
    _closed(line)
    must_fail(lambda: _closed(line[: -len(RESET)]), "an unterminated span")
    bare = klog.Formatter(colour=False).format(rec)
    must_fail(lambda: _distinct(bare), "fields left plain")


def test_hl_follows_the_console(monkeypatch, tmp_path):
    assert klog.hl("9.6 GiB") == "9.6 GiB"
    klog.setup("INFO", stream=io.StringIO())
    assert klog.hl("ckpt/a", "path") == "ckpt/a" and klog.hl("2 s") == "2 s"
    monkeypatch.setenv("FORCE_COLOR", "1")
    console = io.StringIO()
    klog.setup("INFO", stream=console, path=tmp_path / "run.log")
    assert klog.hl("9.6 GiB") == f"\033[93m9.6 GiB{RESET}"
    assert klog.hl("ckpt/a", "path") == f"\033[4;34mckpt/a{RESET}"
    assert klog.hl("cpu") == f"\033[1mcpu{RESET}" and klog.hl("2 s") != "2 s"
    log = klog.get_logger("kohakumoss.hl")
    log.warning(f"slow step {klog.hl('3.5 s')} seen", secs=3.5)
    shown = console.getvalue()
    tint = klog.TINTS["WARNING"]
    assert f"{tint}slow step \033[93m3.5 s{RESET}{tint} seen{RESET}" in shown
    _closed(shown.strip())
    written = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "slow step 3.5 s seen [secs=3.5]" in written
    _plain(written)
    must_fail(lambda: _plain(klog.hl("9.6 GiB")), "hl plain on a colour console")


class Tty(io.StringIO):
    def isatty(self):
        return True


def _colour(stream) -> None:
    assert klog.supports_colour(stream)


CHILD = (
    "from kohakumoss.logging.log import get_logger, hl, setup\n"
    "setup('DEBUG')\n"
    "log = get_logger('kohakumoss.inference.pipeline')\n"
    "log.debug('bound', config='BF=256')\n"
    'log.info(f\'loaded {hl("ckpt/x.safetensors", "path")} in {hl("27.1s")}\', gib=9.63)\n'
    "log.warning('over the plan', peak_gib=10.4)\n"
    "log.error('build failed', op='rms_norm_mod')\n"
)


def _child_stderr(**env: str) -> str:
    """Stderr of a fresh interpreter that logs one line per level; stderr is a pipe."""
    clean = {
        k: v for k, v in os.environ.items() if k not in ("FORCE_COLOR", "NO_COLOR")
    }
    out = subprocess.run(
        [sys.executable, "-c", CHILD],
        env={**clean, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stderr


def _coloured_levels(text: str) -> None:
    lines = text.strip().splitlines()
    assert len(lines) == 4, text
    for line, (_, seq) in zip(lines, LEVEL_CODES[:4]):
        assert seq in line and "[\033[90m" in line, repr(line)
        _distinct(line)
        _closed(line)


def test_force_colour_on_a_pipe():
    forced = _child_stderr(FORCE_COLOR="1", NO_COLOR="1")
    _coloured_levels(forced)
    assert "\033[4;34mckpt/x.safetensors\033[0m" in forced
    assert "] [\033[35minference.pipeline\033[0m] " in forced
    piped = _child_stderr()
    assert "[inference.pipeline] [INFO] loaded ckpt/x.safetensors in 27.1s" in piped
    _plain(piped)
    must_fail(lambda: _coloured_levels(piped), "a pipe coloured without FORCE_COLOR")
    must_fail(lambda: _plain(forced), "FORCE_COLOR ignored")


def test_colour_switches(monkeypatch):
    monkeypatch.setattr(klog.sys, "platform", "linux")
    assert klog.supports_colour(Tty()) and not klog.supports_colour(io.StringIO())
    monkeypatch.setenv("NO_COLOR", "1")
    assert not klog.supports_colour(Tty()) and not klog.supports_colour(io.StringIO())
    must_fail(lambda: _colour(Tty()), "NO_COLOR ignored")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert klog.supports_colour(io.StringIO()) and klog.supports_colour(Tty())
    monkeypatch.setenv("FORCE_COLOR", "0")
    must_fail(lambda: _colour(io.StringIO()), "FORCE_COLOR=0 taken as forcing")


def test_no_colour_off_a_tty(monkeypatch):
    monkeypatch.setattr(klog.sys, "platform", "linux")
    buf = io.StringIO()
    klog.setup("INFO", stream=buf)
    klog.get_logger("kohakumoss.c").warning("piped")
    _plain(buf.getvalue())
    assert "[WARNING] piped" in buf.getvalue()
    tty = Tty()
    klog.setup("INFO", stream=tty)
    klog.get_logger("kohakumoss.c").warning("on a terminal")
    assert tty.getvalue().startswith("[\033[90m")
    must_fail(lambda: _plain(tty.getvalue()), "a terminal left uncoloured")


def _here(rec: logging.LogRecord) -> None:
    assert rec.pathname == __file__


def test_fields_and_caller(collect):
    log = klog.get_logger("kohakumoss.test.fields")
    line = inspect.currentframe().f_lineno + 1
    log.info("loaded", gib=1.5, name="clash")
    rec = collect.records[-1]
    out = klog.Formatter(colour=False).format(rec)
    assert out.endswith("loaded [gib=1.5, name_=clash]"), out
    assert (rec.pathname, rec.lineno) == (__file__, line), (rec.pathname, rec.lineno)
    with km_log.timed(log, "phase") as done:
        done["n"] = 2
    assert collect.records[-1].pathname == __file__
    assert collect.messages[-1] == "phase: done" and collect.records[-1].n == 2
    log.info("again", stacklevel=0)
    must_fail(lambda: _here(collect.records[-1]), "caller frame lost")


def _ours(name: str) -> list[logging.Handler]:
    return [h for h in logging.getLogger(name).handlers if getattr(h, klog.MARK, 0)]


def _one_each() -> None:
    for name in klog.ROOTS:
        assert len(_ours(name)) == 1, (name, logging.getLogger(name).handlers)


def test_setup_is_idempotent():
    buf = io.StringIO()
    klog.setup("INFO", stream=buf)
    klog.setup("INFO", stream=buf)
    _one_each()
    for name in klog.ROOTS:
        assert not logging.getLogger(name).propagate
    klog.get_logger("kohakumoss_data.x").info("one line")
    assert buf.getvalue().count("one line") == 1
    stale = klog.FlushingHandler(buf)
    setattr(stale, klog.MARK, True)
    logging.getLogger("kohakumoss").addHandler(stale)
    must_fail(_one_each, "a second setup that kept the first handler")


def _stdlib(logger: logging.Logger) -> None:
    assert type(logger) is logging.Logger, type(logger)


def test_stdlib_loggers_keep_their_class():
    log = klog.get_logger("kohakumoss.cls")
    log.info("wrapped")
    _stdlib(logging.getLogger("kohakumoss.cls"))
    assert log is klog.get_logger("kohakumoss.cls")
    swapped = logging.getLogger("kohakumoss.cls.swapped")
    swapped.__class__ = type("Swapped", (logging.Logger,), {})
    must_fail(lambda: _stdlib(swapped), "a logger whose class was swapped")


def test_rank_filter():
    buf = io.StringIO()
    klog.setup("INFO", stream=buf, rank=1, world=2)
    klog.get_logger("kohakumoss.r").info("from rank one")
    assert buf.getvalue() == ""
    lead = io.StringIO()
    klog.setup("INFO", stream=lead, rank=0, world=2)
    klog.get_logger("kohakumoss.r").info("from rank zero")
    assert "[rank0] [r] [INFO] from rank zero" in lead.getvalue()
    single = io.StringIO()
    klog.setup("INFO", stream=single)
    klog.get_logger("kohakumoss.r").info("alone")
    assert "[rank" not in single.getvalue() and "alone" in single.getvalue()
    must_fail(lambda: _equal(single.getvalue(), ""), "rank 0 / single process muted")


def _absent(text: str, needle: str) -> None:
    assert needle not in text


def test_level_knob_and_noise(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setenv(klog.LEVEL_ENV, "WARNING")
    klog.setup(stream=buf)
    log = klog.get_logger("kohakumoss.lvl")
    log.info("hidden")
    log.warning("shown")
    assert "hidden" not in buf.getvalue() and "shown" in buf.getvalue()
    assert logging.getLogger("diffusers").level == logging.ERROR
    klog.set_level("INFO")
    log.info("now shown")
    assert "now shown" in buf.getvalue()
    must_fail(lambda: _absent(buf.getvalue(), "now shown"), "set_level ignored")


def _once(handler: Collect, text: str) -> None:
    assert handler.messages.count(text) == 1, handler.messages


def test_info_once(collect):
    klog.info_once("kohakumoss.once", "only once, unique text 1")
    klog.info_once("kohakumoss.once", "only once, unique text 1")
    _once(collect, "only once, unique text 1")
    log = klog.get_logger("kohakumoss.once")
    log.info("twice")
    log.info("twice")
    must_fail(lambda: _once(collect, "twice"), "plain info repeats")


def test_bars_follow_the_level(collect):
    assert not km_log.bar(total=1).disable
    assert km_log.bar(total=1, enabled=False).disable
    logging.getLogger("kohakumoss").setLevel(logging.WARNING)
    must_fail(lambda: _shown(km_log.bar(total=1)), "a bar above INFO")


def _shown(bar) -> None:
    assert not bar.disable
