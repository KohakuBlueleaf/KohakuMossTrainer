"""Export knobs (model-free): SAVE_FORMAT x SAVE_MERGED x ALGO name the files one
export writes, EXPORT_FORMATS maps onto them, bad values are problems, and a full
finetune of a pre-quantized DiT under BASE_PRECISION auto is refused."""

import pytest

from kohakumoss.memory.apply import stored_base_problems
from kohakumoss.training.exports import (
    export_files,
    export_problems,
    from_export_formats,
    with_export_formats,
)
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu
FILES = {
    ("comfy", False, "lora"): ["run-comfy.safetensors"],
    ("hf", False, "lora"): ["run-lycoris.safetensors"],
    ("both", False, "lora"): ["run-comfy.safetensors", "run-lycoris.safetensors"],
    ("both", True, "lora"): [
        "run-comfy.safetensors",
        "run-lycoris.safetensors",
        "run-merged.safetensors",
        "run-diffusers",
    ],
    ("none", True, "lora"): [],
    ("comfy", False, "full"): ["run-merged.safetensors"],
    ("hf", True, "full"): ["run-diffusers"],
}
MAPPED = {
    ("comfy",): ("comfy", False),
    ("lycoris",): ("hf", False),
    ("comfy", "lycoris"): ("both", False),
    ("merged",): ("comfy", True),
    ("lycoris", "diffusers"): ("hf", True),
    (): ("none", False),
}


def _names(save, merged, algo, fused=False) -> list[str]:
    k = {"SAVE_FORMAT": save, "SAVE_MERGED": merged, "ALGO": algo}
    return [f.name("run") for f in export_files(k, fused)]


def check_files(table) -> None:
    for (save, merged, algo), names in table.items():
        assert _names(save, merged, algo) == names, (save, merged, algo)


def test_export_files_per_knobs():
    check_files(FILES)
    fused = export_files({"SAVE_FORMAT": "both", "SAVE_MERGED": True, "ALGO": "lora"}, True)  # fmt: skip
    assert [f.fused for f in fused] == [True, False, True, False]
    swapped = {key: list(reversed(v)) for key, v in FILES.items()}
    must_fail(lambda: check_files(swapped), "the file order reversed")


def test_export_formats_map_onto_save_knobs():
    for formats, want in MAPPED.items():
        assert from_export_formats(formats, full=False) == want, formats
    assert from_export_formats(("merged",), full=True) == ("comfy", False)
    k = with_export_formats(runnable(EXPORT_FORMATS=("lycoris", "merged")))
    assert (k["SAVE_FORMAT"], k["SAVE_MERGED"]) == ("both", True)
    assert with_export_formats(runnable())["SAVE_FORMAT"] == "comfy"
    must_fail(
        lambda: _assert_map(("lycoris",), ("comfy", False)), "lycoris mapped to comfy"
    )


def _assert_map(formats, want) -> None:
    assert from_export_formats(formats, full=False) == want


def test_export_knob_problems():
    assert export_problems(runnable()) == []
    bad = runnable(
        SAVE_FORMAT="diffusers",
        SAVE_FUSED_QKV="yes",
        SAVE_MERGED=1,
        CHECKPOINT_EXPORT="on",
        EXPORT_FORMATS=("onnx",),
    )
    found = " ".join(export_problems(bad))
    for needle in ("SAVE_FORMAT", "SAVE_FUSED_QKV", "SAVE_MERGED", "CHECKPOINT_EXPORT", "onnx"):  # fmt: skip
        assert needle in found, needle
    must_fail(lambda: _assert_clean(bad), "every bad export knob")


def _assert_clean(k) -> None:
    assert export_problems(k) == []


def test_full_finetune_of_a_quantized_dit_is_refused():
    assert stored_base_problems("int8", None, True)
    assert stored_base_problems("int8", "auto", True)
    assert "BASE_PRECISION='compute'" in stored_base_problems("int8", "auto", True)[0]
    assert stored_base_problems("int8", "compute", True) == []
    assert stored_base_problems("int8", None, False) == []
    assert stored_base_problems("compute", None, True) == []
    must_fail(lambda: _assert_none("w4a8", True), "a W4A8 file under full FT")


def _assert_none(stored, full) -> None:
    assert stored_base_problems(stored, "auto", full) == []
