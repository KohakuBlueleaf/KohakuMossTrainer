"""The project's logging: ``[HH:MM:SS] [rankN] [module.name] [LEVEL] message [k=v]``;
modules call :func:`get_logger`, entry points call :func:`setup` once.
See docs/guides/logging.md.
"""

import ctypes
import dataclasses
import logging
import os
import re
import sys
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path, PurePath
from typing import Any

ROOTS = ("kohakumoss", "kohakumoss_kernels", "kohakumoss_data")
LEVEL_ENV = "KOHAKUMOSS_LOG_LEVEL"
FORMAT = "[{time}] {rank}[{name}] [{level}] {message}"
NAME_WIDTH = 26
RESET = "\033[0m"
TIME = "\033[90m"
RANK = "\033[34m"
NAME = "\033[35m"
LEVELS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;37;41m",
}
TINTS = {"WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[31m"}
KEY = "\033[2;36m"
VALUE = "\033[1m"
NUMBER = "\033[93m"
PATH = "\033[4;34m"
HIGHLIGHTS = {"path": PATH, "ok": "\033[92m", "bad": "\033[91m"}
DIM = "\033[2m"
BOOLS = {True: "\033[32m", False: DIM}
HEAD_FIELDS = ("profile", "memory", "name", "kind")
INDENT = "  "
SGR = re.compile(r"\033\[[0-9;]*m")
QUANTITY = re.compile(
    r"[-+]?\d[\d,_]*(\.\d+)?(e[-+]?\d+)?(x|%| ?(GiB|MiB|KiB|GB|MB|ms|us|s|it/s|"
    r"s/step|tokens?|steps?))?",
    re.IGNORECASE,
)
RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
    "rank",
}
NOISY_LIBRARY_LEVELS = {
    "diffusers": logging.ERROR,
    "transformers": logging.ERROR,
    "huggingface_hub": logging.WARNING,
    "urllib3": logging.WARNING,
    "httpx": logging.WARNING,
}
VT_MODE = 7
STD_HANDLES = {"stdout": -11, "stderr": -12}
MARK = "kohakumoss_handler"
COLOUR = "kohakumoss_colour"


def _ours(root: logging.Logger) -> list[logging.Handler]:
    return [h for h in root.handlers if getattr(h, MARK, False)]


def level_of(level: int | str | None) -> int:
    """A logging level from a number, a name, or None (KOHAKUMOSS_LOG_LEVEL, else INFO)."""
    if level is None:
        level = os.environ.get(LEVEL_ENV, "INFO")
    if isinstance(level, str):
        level = level.strip()
        return int(level) if level.isdigit() else getattr(logging, level.upper(), 20)
    return int(level)


def _std_handle(stream: object) -> int | None:
    for name, handle in STD_HANDLES.items():
        if stream is getattr(sys, name, None) or stream is getattr(sys, f"__{name}__"):
            return handle
    return None


def supports_colour(stream: object) -> bool:
    """FORCE_COLOR=1 forces ANSI (over NO_COLOR too), NO_COLOR turns it off; otherwise a
    TTY. On Windows the console of `stream` (stdout or stderr) gets VT processing on."""
    if os.environ.get("FORCE_COLOR", "") not in ("", "0"):
        return True
    if os.environ.get("NO_COLOR"):
        return False
    if not getattr(stream, "isatty", None) or not stream.isatty():
        return False
    if sys.platform != "win32":
        return True
    handle = _std_handle(stream)
    if handle is None:
        return False
    try:
        kernel = ctypes.windll.kernel32
        return bool(kernel.SetConsoleMode(kernel.GetStdHandle(handle), VT_MODE))
    except (AttributeError, OSError):
        return False


def value_colour(value: object) -> str:
    """The SGR of a field value: numbers (with a unit) bright yellow, paths underlined
    blue, anything else bold."""
    if isinstance(value, PurePath):
        return PATH
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return NUMBER
    text = str(value)
    if QUANTITY.fullmatch(text):
        return NUMBER
    if ("/" in text or "\\" in text) and " " not in text:
        return PATH
    return VALUE


def colour_enabled() -> bool:
    """Whether the console handler of the last :func:`setup` writes ANSI."""
    return getattr(logging.getLogger(ROOTS[0]), COLOUR, False)


def hl(text: object, kind: str = "value") -> str:
    """`text` wrapped in its highlight colour (kind value / path / ok / bad; "value"
    picks number or bold like a field) when the console takes colour, else plain."""
    text = str(text)
    if not colour_enabled():
        return text
    code = value_colour(text) if kind == "value" else HIGHLIGHTS[kind]
    return f"{code}{text}{RESET}"


def _paint(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if code and colour_enabled() else text


def _show(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return _paint(BOOLS.get(value, DIM), str(value))
    text = f"{value:.4g}" if isinstance(value, float) else str(value)
    return _paint(value_colour(value), text)


def _same(a: object, b: object) -> bool:
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - an uncomparable value is shown
        return False


def _is_record(obj: object) -> bool:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def _items(obj: object) -> list[tuple[str, object, bool]]:
    """(key, value, equals its default) of a dataclass or a mapping."""
    if _is_record(obj):
        out = []
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if f.default is not dataclasses.MISSING:
                default = f.default
            elif f.default_factory is not dataclasses.MISSING:
                default = f.default_factory()
            else:
                default = dataclasses.MISSING
            out.append((f.name, value, _same(value, default)))
        return out
    if isinstance(obj, Mapping):
        return [(str(k), v, False) for k, v in obj.items()]
    return [("value", obj, False)]


def render(
    obj: object,
    only: Sequence[str] | None = None,
    defaults: bool = True,
    depth: int = 1,
) -> list[str]:
    """Aligned `key  value` lines of a dataclass or a mapping, nested ones indented;
    `only` keeps those keys; `defaults=False` hides fields equal to their default and
    ends with an "(n defaults hidden)" line."""
    items = [i for i in _items(obj) if only is None or i[0] in only]
    shown = [i for i in items if defaults or not i[2]]
    hidden = len(items) - len(shown)
    width = max((len(k) for k, _, _ in shown), default=0)
    pad = INDENT * depth
    lines = []
    for key, value, _ in shown:
        label = _paint(KEY, key)
        if _is_record(value) or (isinstance(value, Mapping) and value):
            lines.append(f"{pad}{label}")
            lines += render(value, None, defaults, depth + 1)
        else:
            gap = " " * (width - len(key) + 2)
            lines.append(f"{pad}{label}{gap}{_show(value)}")
    if hidden:
        lines.append(pad + _paint(DIM, f"({hidden} defaults hidden)"))
    return lines


def headline(obj: object) -> str:
    """The value that names a record (its profile / memory / name / kind field)."""
    for key, value, _ in _items(obj):
        if key in HEAD_FIELDS:
            return _show(value)
    return type(obj).__name__


def log_object(
    log,
    title: str,
    obj: object,
    only: Sequence[str] | None = None,
    suffix: str = "",
    level: int = logging.INFO,
) -> None:
    """`title: <headline><suffix>` then the aligned block of `obj` on `log` (a
    :func:`get_logger` logger); fields equal to their default show only while DEBUG is
    enabled."""
    lines = render(obj, only, defaults=log.isEnabledFor(logging.DEBUG))
    head = f"{title}: {headline(obj)}{suffix}"
    log.log(level, "\n".join([head, *lines]), stacklevel=2)


def _numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _expand(row: Sequence[object]) -> list[list[object]]:
    """A row as display lines: a list cell runs down one line per item, the other
    cells are blank below the row's first line."""
    cols = [v if isinstance(v, list) else [v] for v in row]
    height = max([1, *(len(c) for c in cols)])
    return [[c[j] if j < len(c) else "" for c in cols] for j in range(height)]


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    """An aligned ASCII table: the header row, a `-+-` divider, one line per row (a
    list cell spans one line per item); numeric columns right-aligned."""
    lines = [ln for row in rows for ln in _expand(row)]
    cells = [[_show(v) for v in row] for row in lines]
    plain = [[SGR.sub("", c) for c in row] for row in cells]
    widths = [
        max([len(h)] + [len(row[i]) for row in plain]) for i, h in enumerate(headers)
    ]
    right = [
        any(_numeric(row[i]) for row in lines)
        and all(_numeric(row[i]) or row[i] == "" for row in lines)
        for i in range(len(headers))
    ]

    def line(row: list[str], raw: Sequence[str], align: list[bool]) -> str:
        parts = []
        for i, (cell, text) in enumerate(zip(row, raw)):
            gap = " " * (widths[i] - len(text))
            parts.append(gap + cell if align[i] else cell + gap)
        return (INDENT + " | ".join(parts)).rstrip()

    head = line([_paint(KEY, h) for h in headers], headers, [False] * len(headers))
    divider = INDENT + "-+-".join("-" * w for w in widths)
    return [head, divider, *(line(c, p, right) for c, p in zip(cells, plain))]


def log_table(
    log,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    level: int = logging.INFO,
) -> None:
    """`title` then an aligned table of `rows` under `headers` on `log`."""
    log.log(level, "\n".join([title, *table(headers, rows)]), stacklevel=2)


class Formatter(logging.Formatter):
    """One line per record: time, rank (when set), module name without its package root
    (at most NAME_WIDTH chars), level, message, the caller's fields; coloured when
    `colour`, otherwise plain with any :func:`hl` span stripped."""

    def __init__(self, colour: bool = True) -> None:
        super().__init__()
        self.colour = colour

    def paint(self, code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if self.colour and code else text

    def field(self, key: str, value: object) -> str:
        text = f"{value:.4g}" if isinstance(value, float) else str(value)
        if not self.colour:
            return f"{key}={SGR.sub('', text)}"
        return f"{self.paint(KEY, key)}={self.paint(value_colour(value), text)}"

    def format(self, record: logging.LogRecord) -> str:
        root, _, rest = record.name.partition(".")
        name = rest if rest and root in ROOTS else record.name
        if len(name) > NAME_WIDTH:
            name = "..." + name[-(NAME_WIDTH - 3) :]
        level = record.levelname
        message = record.getMessage()
        tint = TINTS.get(level) if self.colour else None
        if not self.colour:
            message = SGR.sub("", message)
        elif tint:
            message = f"{tint}{message.replace(RESET, RESET + tint)}{RESET}"
        fields = [
            self.field(key, value)
            for key, value in record.__dict__.items()
            if key not in RESERVED
        ]
        if fields:
            message = f"{message} [{', '.join(fields)}]"
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        rank = getattr(record, "rank", None)
        return FORMAT.format(
            time=self.paint(TIME, self.formatTime(record, "%H:%M:%S")),
            rank="" if rank is None else f"[{self.paint(RANK, f'rank{rank}')}] ",
            name=self.paint(NAME, name),
            level=self.paint(LEVELS.get(level, ""), level),
            message=message,
        )


class FlushingHandler(logging.StreamHandler):
    """A stream handler that flushes every record and survives a stream whose encoding
    cannot carry it (a legacy-code-page console gets replacement characters)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record) + self.terminator
            try:
                self.stream.write(line)
            except UnicodeEncodeError:
                encoding = getattr(self.stream, "encoding", None) or "ascii"
                self.stream.write(line.encode(encoding, "replace").decode(encoding))
            self.flush()
        except RecursionError:
            raise
        except Exception:  # noqa: BLE001 - logging must never take the program down
            self.handleError(record)


class RankFilter(logging.Filter):
    """Tags records with their rank; only rank 0 writes."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        return self.rank == 0


def setup(
    level: int | str | None = None,
    stream: object | None = None,
    path: Path | str | None = None,
    rank: int | None = None,
    world: int = 1,
) -> None:
    """Attach the project's handlers to every package root; safe to call again (the
    previous handlers are replaced). `level` defaults to KOHAKUMOSS_LOG_LEVEL, `stream`
    to stderr; `path` adds a plain file handler; with `world` > 1 lines carry `[rankN]`
    and only rank 0 writes."""
    level = level_of(level)
    roots = [logging.getLogger(name) for name in ROOTS]
    old = {h for root in roots for h in _ours(root)}
    for root in roots:
        for handler in _ours(root):
            root.removeHandler(handler)
    for handler in old:
        handler.close()
    target = stream if stream is not None else sys.stderr
    colour = supports_colour(target)
    console = FlushingHandler(target)
    console.setFormatter(Formatter(colour))
    handlers: list[logging.Handler] = [console]
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file = logging.FileHandler(path, encoding="utf-8")
        file.setFormatter(Formatter(False))
        handlers.append(file)
    for handler in handlers:
        setattr(handler, MARK, True)
        handler.setLevel(level)
        if world > 1 and rank is not None:
            handler.addFilter(RankFilter(rank))
    for root in roots:
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
    setattr(roots[0], COLOUR, colour)
    quiet_libraries(level)


def add_log_file(path: Path | str) -> Path:
    """Also write every project log line, uncoloured, to `path` (created), at the level
    `setup` set; the next `setup` call replaces it. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    roots = [logging.getLogger(name) for name in ROOTS]
    file = logging.FileHandler(path, encoding="utf-8")
    file.setFormatter(Formatter(False))
    setattr(file, MARK, True)
    file.setLevel(roots[0].level)
    for root in roots:
        root.addHandler(file)
    return path


def quiet_libraries(level: int) -> None:
    """The NOISY_LIBRARY_LEVELS floors, unless the project logs at DEBUG."""
    if level <= logging.DEBUG:
        return
    for name, floor in NOISY_LIBRARY_LEVELS.items():
        logging.getLogger(name).setLevel(floor)


def set_level(level: int | str) -> None:
    """Raise or lower what the project logs, handlers included."""
    level = level_of(level)
    for name in ROOTS:
        root = logging.getLogger(name)
        root.setLevel(level)
        for handler in _ours(root):
            handler.setLevel(level)


class KMossLogger:
    """A module's logger: level methods take structured fields as keyword arguments (a
    field named like a LogRecord attribute gets a `_` suffix) and record the caller's
    frame; every other attribute is the wrapped stdlib `logger`'s."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def __getattr__(self, name: str) -> Any:
        return getattr(self.logger, name)

    def debug(self, msg: object, *args: object, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, msg, args, kwargs)

    def info(self, msg: object, *args: object, **kwargs: Any) -> None:
        self._emit(logging.INFO, msg, args, kwargs)

    def warning(self, msg: object, *args: object, **kwargs: Any) -> None:
        self._emit(logging.WARNING, msg, args, kwargs)

    def error(self, msg: object, *args: object, **kwargs: Any) -> None:
        self._emit(logging.ERROR, msg, args, kwargs)

    def critical(self, msg: object, *args: object, **kwargs: Any) -> None:
        self._emit(logging.CRITICAL, msg, args, kwargs)

    def exception(self, msg: object, *args: object, **kwargs: Any) -> None:
        kwargs.setdefault("exc_info", True)
        self._emit(logging.ERROR, msg, args, kwargs)

    def log(self, level: int, msg: object, *args: object, **kwargs: Any) -> None:
        self._emit(level, msg, args, kwargs)

    def _emit(
        self, level: int, msg: object, args: tuple, kwargs: dict[str, Any]
    ) -> None:
        if not self.logger.isEnabledFor(level):
            return
        # Two wrapper frames sit between the caller and `_log`.
        stacklevel = int(kwargs.pop("stacklevel", 1)) + 2
        exc_info = kwargs.pop("exc_info", None)
        stack_info = bool(kwargs.pop("stack_info", False))
        fields: dict[str, Any] = dict(kwargs.pop("extra", None) or {})
        fields.update(kwargs)
        extra: Mapping[str, Any] = {
            (f"{k}_" if k in RESERVED else k): v for k, v in fields.items()
        }
        self.logger._log(
            level, msg, args, exc_info, extra or None, stack_info, stacklevel
        )


_LOGGERS: dict[str, KMossLogger] = {}


def get_logger(name: str) -> KMossLogger:
    """The logger of a module (`__name__`); a name outside the package roots is put
    under "kohakumoss"."""
    if name.split(".", 1)[0] not in ROOTS:
        name = f"kohakumoss.{name}"
    if name not in _LOGGERS:
        _LOGGERS[name] = KMossLogger(logging.getLogger(name))
    return _LOGGERS[name]


@cache
def debug_once(name: str, msg: str) -> None:
    """Log `msg` at DEBUG on logger `name` the first time only."""
    get_logger(name).debug(msg)


@cache
def info_once(name: str, msg: str) -> None:
    """Log `msg` at INFO on logger `name` the first time only."""
    get_logger(name).info(msg)


@cache
def warning_once(name: str, msg: str) -> None:
    """Log `msg` at WARNING on logger `name` the first time only."""
    get_logger(name).warning(msg)
