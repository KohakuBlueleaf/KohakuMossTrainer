"""One default for an empty RUN_NAME: the output folder and the export files agree."""

from types import SimpleNamespace

import pytest

from kohakumoss.training.build.model import export
from kohakumoss.training.knobs import run_name
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


class _Spy:
    def __init__(self):
        self.paths = []

    def export(self, path, fmt, metadata, row_scales=None, fused_qkv=False):
        self.paths.append(path)


def _names(k, tmp_path) -> list[str]:
    spy = _Spy()
    built = SimpleNamespace(backbone=SimpleNamespace(fold_scales=None), trainable=spy)
    export(k, built, tmp_path, {})
    return [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in spy.paths]


def _prefixed(names: list[str], prefix: str) -> None:
    assert names and all(n.startswith(prefix + "-") for n in names), names


def test_empty_run_name_is_one_default(tmp_path):
    k = {
        "RUN_NAME": "",
        "ALGO": "lora",
        "SAVE_FORMAT": "both",
        "SAVE_MERGED": False,
        "SAVE_FUSED_QKV": False,
    }
    names = _names(k, tmp_path)
    _prefixed(names, run_name(k))
    assert run_name(k) == "run"
    must_fail(lambda: _prefixed(names, "moss"), "export default differs")
