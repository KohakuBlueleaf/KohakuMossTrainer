"""A training state is read only with its own tag / spec / fold metadata: export refuses
a file without it with one message naming what is missing and writes nothing; a
complete state passes the check; an untagged file inside a run checkpoint is refused
as an adapter file, any other untagged file loads as one."""

import pytest
import torch
from safetensors.torch import save_file

from kohakumoss.adapters.export.cli import export_main
from kohakumoss.adapters.train.adapter_set import (
    FOLD_KEY,
    FULL_TAG,
    RAW_TAG,
    SPEC_KEY,
    check_adapter_file,
    state_kind,
)
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
COMPLETE = {RAW_TAG: "1", SPEC_KEY: "{}", FOLD_KEY: "null"}
NO_TAG = (
    "no training-state metadata (kohakumoss_adapter_state or kohakumoss_full_state); "
    "re-export from a run of this version"
)


def refused(meta: dict, why: str) -> None:
    try:
        state_kind("state.safetensors", meta)
    except ValueError as err:
        assert str(err) == f"state.safetensors: {why}", err
        return
    raise AssertionError(f"{meta} was accepted")


def test_metadata_is_required():
    refused({}, NO_TAG)
    without = {RAW_TAG: "1", SPEC_KEY: "{}"}
    why = "training-state metadata without kohakumoss_fold; "
    refused(without, why + "re-export from a run of this version")
    assert state_kind("s", COMPLETE) == RAW_TAG
    assert state_kind("s", {FULL_TAG: "1", FOLD_KEY: "null"}) == FULL_TAG
    must_fail(lambda: refused(COMPLETE, NO_TAG), "a complete training state")


def _run_file_refused(check, path) -> None:
    try:
        check(path, {})
    except ValueError as err:
        assert str(err) == f"{path}: {NO_TAG}", err
        return
    raise AssertionError(f"untagged {path} inside a run was accepted")


def test_an_untagged_run_checkpoint_is_refused(tmp_path):
    step = tmp_path / "run" / "checkpoints" / "step_00000100"
    step.mkdir(parents=True)
    (tmp_path / "run" / "config.json").write_text("{}", encoding="utf-8")
    inside = step / "trainable.safetensors"
    _run_file_refused(check_adapter_file, inside)
    check_adapter_file(tmp_path / "plain_lora.safetensors", {})
    check_adapter_file(tmp_path / "run" / "x" / "y" / "lora.safetensors", {})
    check_adapter_file(inside, COMPLETE)

    def tags_only(path, meta) -> None:
        if any(meta.get(tag) for tag in (RAW_TAG, FULL_TAG)):
            state_kind(path, meta)

    must_fail(lambda: _run_file_refused(tags_only, inside), "the run layout ignored")


def test_export_refuses_and_writes_nothing(tmp_path):
    state = tmp_path / "trainable.safetensors"
    save_file({"lycoris_x.lora_down.weight": torch.zeros(1, 1)}, str(state), {})
    out = tmp_path / "out.safetensors"
    cfg = {"WEIGHTS": str(tmp_path / "w"), "ADAPTER": str(state), "OUT": str(out)}
    with pytest.raises(SystemExit) as caught:
        export_main(cfg)
    assert str(caught.value) == f"export: {state}: {NO_TAG}"
    assert not out.exists()
