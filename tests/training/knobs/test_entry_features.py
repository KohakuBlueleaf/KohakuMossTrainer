"""The doctor report: OS, packages, devices, kernels and memory plans in one JSON. The
kernel matrix written is the one the registry reports for each device kind, every op by
every backend, and every memory profile is resolved."""

import json
import sys

import pytest

from kohakumoss.cli import doctor
from kohakumoss.memory.profiles import ORDER
from kohakumoss_kernels import ops as kernel_ops
from kohakumoss_kernels import registry
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.slow]


def check_report(rep: dict) -> None:
    assert {"os", "python", "packages", "devices", "kernels", "memory_plans"} <= set(
        rep
    )
    assert set(doctor.PACKAGES) <= set(rep["packages"]), rep["packages"]
    kinds = doctor.device_kinds()
    assert list(rep["kernels"]) == kinds, (list(rep["kernels"]), kinds)
    for dev in kinds:
        tables = rep["kernels"][dev]
        assert set(tables) == {"float16", "bfloat16"}, (dev, set(tables))
        for dtype, table in tables.items():
            assert set(table) == set(kernel_ops.ALL), (dev, dtype)
            for op, row in table.items():
                assert set(row) == set(registry.BACKENDS), (dev, dtype, op)
                assert all(isinstance(v, bool) for v in row.values()), row
        assert tables == doctor.kernel_matrix(dev), dev
    assert list(rep["memory_plans"]) == list(ORDER), list(rep["memory_plans"])
    assert all(rep["memory_plans"].values()), rep["memory_plans"]


def test_doctor_json(tmp_path, monkeypatch):
    out = tmp_path / "doctor.json"
    monkeypatch.setattr(
        sys, "argv", ["kohakumoss-doctor", "--json", str(out), "--memory"]
    )
    doctor.main()
    rep = json.loads(out.read_text(encoding="utf-8"))
    check_report(rep)
    first = next(iter(kernel_ops.ALL))
    dropped = json.loads(json.dumps(rep))
    for tables in dropped["kernels"].values():
        tables["float16"].pop(first)
    must_fail(lambda: check_report(dropped), "an op missing from the matrix")
    flipped = json.loads(json.dumps(rep))
    row = flipped["kernels"]["cpu"]["bfloat16"][first]
    backend = next(iter(row))
    row[backend] = not row[backend]
    must_fail(lambda: check_report(flipped), "a backend's availability flipped")
    short = json.loads(json.dumps(rep))
    short["memory_plans"].pop(next(iter(ORDER)))
    must_fail(lambda: check_report(short), "a memory profile missing")
