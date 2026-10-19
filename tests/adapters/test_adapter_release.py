"""Adapted layers are freed by reference counting alone, model-free: with the cyclic
collector off, `del` kills the model, the adapter set, an adapted linear, its LyCORIS
module and a factor, for every kind on one real linear (dense and int8 bases, the
bound forward run once) and for LoRA with module dropout (a train/eval re-bind hook)
on the MOSS-Image base architecture on the meta device; likewise a model with an
inference AdapterStack loaded. A strong `org_module` / `org_forward` back to the layer,
or a base forward bound to it in the inference chain, keeps them alive."""

import types
from functools import partial

import pytest
import torch

from kohakumoss.adapters import AdapterConfig, AdapterStack, apply
from kohakumoss.adapters.infer.inference import _lyco_bypass
from kohakumoss.model import build_backbone, resolve_config
from kohakumoss.model.opset import OpSet
from tests._util.negative import must_fail
from tests._util.release import check_released
from tests.adapters.test_stored_bypass import (
    KINDS,
    adapted,
    dense_linear,
    holder,
    perturb,
    problem,
)

pytestmark = pytest.mark.cpu
CPU = torch.device("cpu")


def _strong_org(aset) -> None:
    t = aset.targets[0]
    t.lyco.org_module = [t.linear]


def _strong_org_forward(aset) -> None:
    t = aset.targets[0]
    t.lyco.org_forward = types.MethodType(type(t.linear).forward, t.linear)


def _one_linear(kind: str, base: str, broken=None):
    """`build()` of one adapted linear (`base`: "dense" or a quant format) after one
    forward; `broken(aset)` puts a cycle back."""

    def build() -> tuple:
        w, b, lin, x = problem("int8")
        if base == "dense":
            lin = dense_linear(w, b)
        aset = adapted(lin, kind)
        perturb(aset)
        with torch.no_grad():
            lin(x)
        if broken is not None:
            broken(aset)
        t = aset.targets[0]
        factor = next(t.lyco.parameters())
        return aset.model, aset, lin, t.lyco, factor

    return build


@pytest.mark.parametrize("base", ["dense", "int8"])
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_adapted_linear_freed_on_del(kind, base):
    check_released(_one_linear(kind, base))


def test_a_strong_back_reference_keeps_the_layer():
    for broken in (_strong_org, _strong_org_forward):
        build = _one_linear("lora", "dense", broken)
        must_fail(lambda: check_released(build), broken.__name__)


def _backbone(broken=None):
    """`build()` of LoRA with module dropout on the real architecture (meta weights,
    host adapters), switched to eval and back."""

    def build() -> tuple:
        model = build_backbone(resolve_config("base"), "meta", torch.bfloat16)
        model.bind_ops(OpSet.resolve(CPU, torch.bfloat16, {}))
        config = AdapterConfig(
            algo="lora", dim=4, alpha=4.0, module_dropout=0.1, backend="torch"
        )
        aset = apply(model, config, device=CPU)
        model.eval()
        model.train()
        t = aset.targets[0]
        assert "train" in vars(t.linear), "no re-bind hook installed"
        if broken is not None:
            broken(aset)
        return model, aset, t.linear, t.lyco, next(t.lyco.parameters())

    return build


def test_adapted_backbone_freed_on_del():
    check_released(_backbone())
    must_fail(lambda: check_released(_backbone(_strong_org)), "a strong org_module")


def _stacked(bound_base: bool = False):
    """`build()` of a fresh linear with a LoRA file's tensors loaded in bypass."""

    def build() -> tuple:
        w, b, lin, x = problem("int8")
        source = adapted(dense_linear(w, b), "lora")
        perturb(source)
        tensors = source.tensors("lycoris")
        fresh = dense_linear(w, b)
        root = holder(fresh)
        stack = AdapterStack(root, mode="bypass")
        stack.add(tensors, 1.0)
        with torch.no_grad():
            fresh(x)
        if bound_base:
            lyco = stack.live[0]
            prev = types.MethodType(type(fresh).forward, fresh)
            vars(fresh)["forward"] = partial(_lyco_bypass, prev, lyco, 1.0)
        return root, stack, fresh, stack.live[0]

    return build


def test_inference_stack_freed_on_del():
    check_released(_stacked())
    must_fail(lambda: check_released(_stacked(True)), "a bound base forward")
