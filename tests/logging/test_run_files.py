"""Run files across attempts of one RUN_NAME, model-free: a fresh run moves an earlier
attempt's metrics.jsonl / log.txt aside to the first free numbered name, and a
resumed run keeps the rows up to its step (later rows moved aside) and appends."""

import json
import logging

import pytest

from kohakumoss.logging import sinks
from kohakumoss.logging.sinks import ConsoleLogger, fresh_run
from tests._util.logs import collecting
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _attempt(out, steps) -> None:
    logger = ConsoleLogger(str(out))
    for step in steps:
        logger.log({"loss": float(step)}, step)
    logger.close()


def _steps(path) -> list[int]:
    return [json.loads(x)["step"] for x in path.read_text().splitlines()]


def check_fresh(out, start) -> None:
    _attempt(out, [1, 2, 3])
    (out / "log.txt").write_text("old log\n")
    (out / "metrics.1.jsonl").write_text("")
    with collecting(logging.WARNING) as logs:
        start(out)
    _attempt(out, [1, 2])
    assert _steps(out / "metrics.jsonl") == [1, 2]
    assert _steps(out / "metrics.2.jsonl") == [1, 2, 3]
    assert (out / "log.1.txt").read_text() == "old log\n"
    assert any("fresh run" in m for m in logs.messages), logs.messages


def test_fresh_run_moves_the_earlier_attempt(tmp_path):
    check_fresh(tmp_path / "a", fresh_run)
    must_fail(lambda: check_fresh(tmp_path / "b", lambda out: None), "plain append")


def check_resume(out, trim: bool) -> None:
    _attempt(out, [1, 2, 3, 4, 5])
    logger = ConsoleLogger(str(out))
    if trim:
        logger.resume_at(2)
    for step in (3, 4):
        logger.log({"loss": 0.0}, step)
    logger.close()
    assert _steps(out / "metrics.jsonl") == [1, 2, 3, 4]
    assert _steps(out / "metrics.1.jsonl") == [3, 4, 5]


def test_resume_keeps_rows_up_to_its_step(tmp_path):
    check_resume(tmp_path / "a", True)
    must_fail(lambda: check_resume(tmp_path / "b", False), "rows past the resume kept")


def test_free_name_and_empty_files(tmp_path):
    empty = tmp_path / "metrics.jsonl"
    empty.write_text("")
    assert sinks.rotate(empty) is None and empty.exists()
    assert sinks.rotate(tmp_path / "missing.txt") is None
    assert sinks.free_name(tmp_path / "log.txt").name == "log.1.txt"
