"""Train-script knobs map onto AdapterSpec; invalid combinations are all reported."""

import lycoris.kernels as lyk
import pytest
from _adtoy import toy
from lycoris.kernels.plans import tune

from kohakumoss.adapters import AdapterSpec, apply, spec_from_knobs

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


def _one(spec: AdapterSpec, text: str) -> None:
    problems = spec.validate()
    assert any(text in p for p in problems), problems


def test_tlora_trains_in_bypass_only():
    _one(AdapterSpec(algo="tlora", mode="merge"), "bypass")
    assert not AdapterSpec(algo="tlora").validate()


def test_dora_with_input_dropout_is_rejected():
    _one(AdapterSpec(algo="lora", dora=True, dropout=0.1), "input dropout")
    assert not AdapterSpec(algo="lora", dora=True, rank_dropout=0.1).validate()


def test_algo_kwargs_are_checked_against_the_module():
    _one(AdapterSpec(algo="tlora", algo_kwargs={"alpha": 1.5}), "own knob")
    _one(AdapterSpec(algo="oft", algo_kwargs={"block_size": 32}), "block_size")
    _one(AdapterSpec(algo="lora", algo_kwargs={"use_scaler": True}), "use_scaler")
    ok = {"use_scalar": True, "rs_lora": True}
    assert not AdapterSpec(algo="lora", algo_kwargs=ok).validate()
    assert not AdapterSpec(algo="lokr", algo_kwargs={"factor": 4}).validate()
    assert not AdapterSpec(algo="oft", algo_kwargs={"constraint": 1e-3}).validate()
    _one(AdapterSpec(algo="lora", algo_kwargs={"mask_alpha": 1.0}), "mask_alpha")


def test_seed_and_tune_come_from_the_run_knobs():
    spec = spec_from_knobs({"SEED": 7, "TUNE": "off"})
    assert spec.seed == 7 and spec.tune is False
    assert spec_from_knobs({}).tune is True


def test_tune_off_turns_the_lycoris_race_off():
    apply(toy(), AdapterSpec(preset="attn", dim=2, tune=False))
    assert not lyk.tuning_enabled() and not tune.tuning_enabled()
    apply(toy(), AdapterSpec(preset="attn", dim=2))
    assert lyk.tuning_enabled() and tune.tuning_enabled()
