"""Inference stacks: every format stacks in bypass and merge alike, base-dependent
adapters see the stacked weight, dense deltas live in the compute dtype, one-way merges
keep no copies, and a pinned training backend does not travel with a state file."""

import pytest
import torch
from _adcases import build, make_spec, outputs
from _adtoy import inputs, perturb, toy
from lycoris.kernels import tier

from kohakumoss.adapters import AdapterSpec, apply, inference, load_adapters
from kohakumoss.adapters.backend import pick_backend
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
TOL = dict(rtol=2e-4, atol=2e-5)


def _file(tmp_path, name, tag, fmt="comfy", seed=0, **kw):
    _, aset = build(name, seed=seed, **kw)
    path = tmp_path / f"{tag}.safetensors"
    if fmt == "raw":
        aset.save_state(str(path))
    else:
        aset.export(str(path), fmt, dtype=None)
    return str(path)


def _stacked(paths, mode):
    model = toy()
    model.eval()
    stack = load_adapters(model, [(p, m) for p, m in paths], mode)
    return outputs(model), stack, model


@pytest.mark.parametrize("second", ["dora-out", "doha", "oft", "glora", "ia3"])
def test_second_adapter_sees_the_stacked_weight(second, tmp_path):
    paths = [
        (_file(tmp_path, "lora", "a"), 1.0),
        (_file(tmp_path, second, "b", seed=1), 0.8),
    ]
    bypass, _, _ = _stacked(paths, "bypass")
    merged, _, _ = _stacked(paths, "merge")
    torch.testing.assert_close(bypass, merged, **TOL)


def test_dora_against_the_original_weight_is_caught(monkeypatch, tmp_path):
    paths = [
        (_file(tmp_path, "lora", "a"), 1.0),
        (_file(tmp_path, "dora-out", "b", seed=1), 1.0),
    ]
    merged, _, _ = _stacked(paths, "merge")
    monkeypatch.setattr(inference.AdapterStack, "_unfolded_base", lambda s, m, p: None)

    def original():
        bypass, _, _ = _stacked(paths, "bypass")
        torch.testing.assert_close(bypass, merged, **TOL)

    must_fail(original, "a stacked DoRA built against W0")


def test_raw_state_stacks_after_a_file(tmp_path):
    paths = [
        (_file(tmp_path, "lora", "a"), 1.0),
        (_file(tmp_path, "loha", "b", fmt="raw", seed=1), 0.7),
    ]
    bypass, stack, model = _stacked(paths, "bypass")
    merged, _, _ = _stacked(paths, "merge")
    torch.testing.assert_close(bypass, merged, **TOL)
    stack.remove()
    plain = toy()
    plain.eval()
    torch.testing.assert_close(outputs(model), outputs(plain))


def test_dense_deltas_are_held_in_the_compute_dtype(tmp_path):
    path = _file(tmp_path, "dora-out", "a")
    model = toy(dtype=torch.bfloat16)
    model.eval()
    load_adapters(model, [(path, 1.0)], "bypass")
    layer = model.get_submodule("layers.0.proj.q")
    fn = vars(layer)["forward"]
    assert fn.func is inference._dense_add
    assert fn.args[2].dtype == torch.bfloat16


def test_one_way_merge_keeps_no_copies(tmp_path):
    path = _file(tmp_path, "lora", "a")
    model = toy()
    stack = load_adapters(model, [(path, 1.0)], "merge", undoable=False)
    assert not stack.originals
    with pytest.raises(RuntimeError):
        stack.remove()
    undo = load_adapters(toy(), [(path, 1.0)], "merge")
    assert undo.originals


def test_loaded_modules_run_on_the_stack_backend(monkeypatch, tmp_path):
    path = _file(tmp_path, "lora", "a")
    seen = []
    real = tier.run

    def spy(op, tensors, supported, backend, *rest):
        seen.append(backend)
        return real(op, tensors, supported, backend, *rest)

    monkeypatch.setattr(tier, "run", spy)
    model = toy()
    model.eval()
    stack = load_adapters(model, [(path, 1.0)], "bypass")
    assert all(lyco.kernel_backend == stack.backend for lyco in stack.live)
    outputs(model)
    assert seen and all(b == stack.backend for b in seen), set(seen)


def test_a_pinned_training_backend_does_not_travel(tmp_path):
    with pytest.raises(ValueError, match="CUDA"):
        apply(toy(), AdapterSpec(preset="attn", dim=2, backend="triton"))
    _, aset = build("lora")
    aset.spec = AdapterSpec(**{**aset.spec.__dict__, "backend": "triton"})
    path = tmp_path / "raw.safetensors"
    aset.save_state(str(path))
    model = toy()
    model.eval()
    load_adapters(model, [(str(path), 1.0)], "bypass")
    assert torch.isfinite(outputs(model)).all()
    assert pick_backend("cpu", "compile") == "compile"
    assert pick_backend("cpu", "auto") == "torch"


def _tier_backends(monkeypatch, model, aset) -> set:
    seen = set()
    real = tier.run

    def spy(op, tensors, supported, backend, *rest):
        seen.add(backend)
        return real(op, tensors, supported, backend, *rest)

    monkeypatch.setattr(tier, "run", spy)
    model(*inputs()).sum().backward()
    return seen


def test_eager_body_layers_take_the_bound_tier(monkeypatch):
    model = toy()
    aset = apply(model, make_spec("lora", dropout=0.5, backend="torch"))
    perturb(aset)
    model.train()
    seen = _tier_backends(monkeypatch, model, aset)
    assert seen == {"torch"}, seen

    def unbound():
        for t in aset.targets:
            t.lyco.kernel_backend = None
        assert _tier_backends(monkeypatch, model, aset) == {"torch"}

    must_fail(unbound, "eager-body layers dispatching per call")
