"""Train-script knobs map onto AdapterSpec; invalid combinations are all reported."""

import pytest

from kohakumoss.adapters import AdapterSpec, spec_from_knobs

pytestmark = pytest.mark.cpu


def test_knobs_map_onto_spec():
    spec = spec_from_knobs(
        {
            "ALGO": "lokr",
            "PRESET": "attn",
            "DIM": 8,
            "ALPHA": 4.0,
            "ALGO_KWARGS": {"factor": 8, "wd_on_out": False},
            "DORA": True,
            "DROPOUT": {"input": 0.1, "rank": 0.2, "module": 0.05},
            "ADAPTER_MODE": "merge",
            "TRAIN_TEXT_REFINER": True,
            "KERNELS": {"lycoris": "tilelang"},
        }
    )
    assert spec.lycoris_algo == "lokr" and spec.preset == "attn+text-refiner"
    assert spec.dora and not spec.dora_on_output
    assert (spec.dropout, spec.rank_dropout, spec.module_dropout) == (0.1, 0.2, 0.05)
    assert spec.backend == "tilelang" and spec.mode == "merge"
    kwargs = spec.module_kwargs()
    assert (
        kwargs["factor"] == 8 and kwargs["weight_decompose"] and not kwargs["wd_on_out"]
    )
    assert kwargs["bypass_mode"] is False


def test_every_problem_is_listed():
    spec = AdapterSpec(algo="nope", mode="sideways", dim=0, dropout=1.5)
    problems = spec.validate()
    assert len(problems) == 4


def test_dora_on_oft_and_full_bypass_are_rejected():
    assert AdapterSpec(algo="oft", dora=True).validate()
    assert AdapterSpec(algo="full", mode="bypass").validate()
    assert not AdapterSpec(algo="full", mode="merge").validate()
