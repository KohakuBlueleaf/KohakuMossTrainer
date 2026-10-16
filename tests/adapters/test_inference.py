"""Loaded adapters reproduce the training-time forward; merge and remove are exact."""

import pytest
import torch
from _adcases import CASES, build, outputs
from _adtoy import inputs, toy

from kohakumoss.adapters import load_adapters
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
TOL = dict(rtol=3e-4, atol=3e-5)
FILE_FORMATS = ("lycoris", "comfy", "comfy-dm", "raw")
LINEAR_IN_STRENGTH = {"oft-constraint-rescale"}
LOSSLESS_NATIVE = {"lora", "lora-scalar", "lora-rs", "loha", "lokr", "lokr-both", "lokr-full",
                   "oft", "oft-constraint", "oft-constraint-rescale", "boft", "glora", "full",
                   "dora-out", "dora-in", "doha", "dokr", "lora-norms", "lora-text",
                   "ia3", "ia3-in", "tlora"}  # fmt: skip


def _export(aset, fmt, path):
    match fmt:
        case "lycoris":
            aset.export(str(path), "lycoris", dtype=None)
        case "comfy":
            aset.export(str(path), "comfy", dtype=None)
        case "comfy-dm":
            aset.export(str(path), "comfy", style="diffusion_model", dtype=None)
        case "raw":
            aset.save_state(str(path))


def _cases():
    for name in sorted(CASES):
        for fmt in FILE_FORMATS:
            if fmt == "lycoris" and name not in LOSSLESS_NATIVE:
                continue
            yield name, fmt


@pytest.mark.parametrize("mode", ["bypass", "merge"])
@pytest.mark.parametrize("name,fmt", list(_cases()))
def test_loaded_adapter_reproduces_training_forward(name, fmt, mode, tmp_path):
    model, aset = build(name)
    linear_only = name in LINEAR_IN_STRENGTH and fmt.startswith("comfy")
    m = 1.0 if linear_only else 0.7
    aset.set_multiplier(m)
    want = outputs(model)
    path = tmp_path / "a.safetensors"
    _export(aset, fmt, path)
    fresh = toy()
    fresh.eval()
    base = outputs(fresh)
    stack = load_adapters(fresh, [(str(path), m)], mode=mode)
    torch.testing.assert_close(outputs(fresh), want, **TOL)
    stack.remove()
    assert torch.equal(outputs(fresh), base)


def test_two_adapters_stack_with_their_own_multipliers(tmp_path):
    m1, a1 = build("lora", seed=0)
    _, a2 = build("loha", seed=0)
    p1, p2 = tmp_path / "1.safetensors", tmp_path / "2.safetensors"
    a1.export(str(p1), "comfy", dtype=None)
    a2.export(str(p2), "comfy", dtype=None)
    merged = toy()
    merged.eval()
    stack = load_adapters(merged, [(str(p1), 0.5), (str(p2), 1.5)], mode="merge")
    bypass = toy()
    bypass.eval()
    load_adapters(bypass, [(str(p1), 0.5), (str(p2), 1.5)], mode="bypass")
    torch.testing.assert_close(outputs(merged), outputs(bypass), **TOL)

    def one_only():
        single = toy()
        single.eval()
        load_adapters(single, [(str(p1), 0.5)], mode="merge")
        torch.testing.assert_close(outputs(single), outputs(merged), **TOL)

    must_fail(one_only, "the second adapter must contribute")
    stack.remove()


def test_unmerge_restores_weights_bit_exactly(tmp_path):
    _, aset = build("dora-out")
    path = tmp_path / "a.safetensors"
    aset.export(str(path), "comfy")
    model = toy()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    stack = load_adapters(model, [(str(path), 1.0)], mode="merge")
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    stack.unmerge()
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_inference_saves_no_tensors_for_backward(tmp_path):
    _, aset = build("lora")
    path = tmp_path / "a.safetensors"
    aset.export(str(path), "comfy")
    model = toy()
    model.eval()
    load_adapters(model, [(str(path), 1.0)], mode="bypass")
    latent, text = inputs()
    with torch.no_grad():
        y = model(latent, text)
    assert y.grad_fn is None
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("mode", ["bypass", "merge"])
@pytest.mark.parametrize("name", ["lora", "lokr", "oft", "glora", "dora-out", "ia3"])
def test_row_scales_put_unfolded_files_on_a_folded_model(name, mode, tmp_path):
    _, aset = build(name)
    path = tmp_path / "a.safetensors"
    aset.export(str(path), "comfy", dtype=None)
    ref = toy()
    load_adapters(ref, [(str(path), 1.0)], mode=mode)
    folded = toy()
    layer = "layers.0.proj.q"
    scale = torch.linspace(0.5, 2.0, 8).repeat_interleave(4)
    with torch.no_grad():
        folded.get_submodule(layer).weight.mul_(scale[:, None])
    load_adapters(folded, [(str(path), 1.0)], mode=mode, row_scales={layer: scale})
    x = torch.randn(5, 32)
    with torch.no_grad():
        want = ref.get_submodule(layer)(x) * scale
        got = folded.get_submodule(layer)(x)
    torch.testing.assert_close(got, want, **TOL)


def test_unknown_keys_are_an_error(tmp_path):
    model = toy()
    with pytest.raises(KeyError):
        load_adapters(
            model, [({"lora_unet_nowhere.lora_up.weight": torch.zeros(1, 1)}, 1.0)]
        )
