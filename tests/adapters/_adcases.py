"""Adapter configurations every adapter test sweeps."""

import torch
from _adref import lycoris_delta
from _adtoy import inputs, perturb, toy

from kohakumoss.adapters import AdapterSpec, apply

CASES = {
    "lora": dict(algo="lora"),
    "lora-scalar": dict(algo="lora", algo_kwargs={"use_scalar": True}),
    "lora-rs": dict(algo="lora", algo_kwargs={"rs_lora": True}),
    "loha": dict(algo="loha"),
    "lokr": dict(algo="lokr", dim=2, algo_kwargs={"factor": 4}),
    "lokr-both": dict(
        algo="lokr", dim=2, algo_kwargs={"factor": 4, "decompose_both": True}
    ),
    "lokr-full": dict(algo="lokr", dim=64, algo_kwargs={"factor": 4}),
    "oft": dict(algo="oft", dim=4),
    "oft-constraint": dict(algo="oft", dim=4, algo_kwargs={"constraint": 1e-3}),
    "oft-constraint-rescale": dict(
        algo="oft", dim=4, algo_kwargs={"constraint": 1e-3, "rescaled": True}
    ),
    "boft": dict(algo="boft", dim=4, preset="attn"),
    "glora": dict(algo="glora"),
    "ia3": dict(algo="ia3"),
    "ia3-in": dict(algo="ia3", algo_kwargs={"train_on_input": True}),
    "dylora": dict(algo="dylora", algo_kwargs={"block_size": 2}),
    "tlora": dict(algo="tlora"),
    "full": dict(algo="full", mode="merge"),
    "dora-out": dict(algo="lora", dora=True),
    "dora-in": dict(algo="lora", dora=True, dora_on_output=False),
    "doha": dict(algo="loha", dora=True),
    "dokr": dict(algo="lokr", dim=2, dora=True, algo_kwargs={"factor": 4}),
    "lora-norms": dict(algo="lora", train_norms=True),
    "lora-text": dict(algo="lora", preset="attn-mlp+text-refiner+modulation"),
}
NATIVE_REF = {
    "locon": "locon",
    "loha": "loha",
    "lokr": "lokr",
    "glora": "glora",
    "oft": "oft",
    "boft": "boft",
    "ia3": "ia3",
    "full": "full",
}


def make_spec(name: str, **override) -> AdapterSpec:
    kw = dict(preset="attn-mlp", dim=4, alpha=2.0)
    kw.update(CASES[name])
    kw.update(override)
    return AdapterSpec(**kw)


def build(name: str, seed: int = 0, **override):
    """(model, adapter set) with every adapter tensor randomised."""
    model = toy(seed)
    aset = apply(model, make_spec(name, **override))
    perturb(aset)
    model.eval()
    return model, aset


def reference_model(aset, seed: int = 0):
    """A plain toy whose weights are W + ΔW from independent formulas over the export."""
    ref = toy(seed)
    modules = dict(ref.named_modules())
    comfy = aset.tensors("comfy", "diffusion_model")
    native = aset.tensors("lycoris")
    with torch.no_grad():
        for t in aset.targets:
            layer = modules[t.path]
            w = layer.weight
            if t.kind in NATIVE_REF:
                parts = _parts(native, t.lora_name)
                dw = lycoris_delta(NATIVE_REF[t.kind], parts, w, w.shape[0])
            else:
                parts = _parts(comfy, f"diffusion_model.{t.path}")
                dw = lycoris_delta("locon", parts, w, w.shape[0])
            layer.weight.add_(dw.to(w.dtype))
            if t.kind == "full" and f"{t.lora_name}.diff_b" in native:
                layer.bias.add_(native[f"{t.lora_name}.diff_b"].to(layer.bias.dtype))
        for n in aset.norms:
            modules[n.path].weight.copy_(n.module.weight.detach())
    ref.eval()
    return ref


def _parts(tensors: dict, stem: str) -> dict:
    prefix = stem + "."
    return {k[len(prefix) :]: v for k, v in tensors.items() if k.startswith(prefix)}


def outputs(model):
    with torch.no_grad():
        return model(*inputs())
