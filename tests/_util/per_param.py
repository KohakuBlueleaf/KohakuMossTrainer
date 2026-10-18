"""PerParamOptimizer (test helper): one torch optimiser per parameter behind a
torch-style `param_groups` view, the per-parameter `step_param` interface the fused
step calls."""

from collections.abc import Callable, Iterable

from torch import Tensor


class PerParamOptimizer:
    """One optimiser per parameter behind a torch-style `param_groups` view (for
    schedules). `params` is an iterable of parameters or of group dicts; a parameter's
    optimiser takes its group's hyperparameters on every step."""

    def __init__(self, params: Iterable, factory: Callable, **hparams):
        items = list(params)
        if not items or not isinstance(items[0], dict):
            items = [{"params": items}]
        self.param_groups, self.params, self.optims = [], [], {}
        self.group_of: dict[Tensor, dict] = {}
        for item in items:
            own = {k: v for k, v in item.items() if k != "params"}
            params = [p for p in item["params"] if p.requires_grad]
            group = None
            for p in params:
                opt = factory([{**own, "params": [p]}], **hparams)
                if group is None:
                    group = dict(opt.param_groups[0], params=params)
                self.optims[p] = opt
                self.group_of[p] = group
            self.params += params
            self.param_groups.append(group or {**hparams, **own, "params": []})
        self._keys = {
            id(g): tuple(k for k in g if k != "params") for g in self.param_groups
        }

    def step_param(self, p: Tensor) -> None:
        opt = self.optims[p]
        shared, group = self.group_of[p], opt.param_groups[0]
        for k in self._keys[id(shared)]:
            group[k] = shared[k]
        opt.step()

    def step(self) -> None:
        for p in self.params:
            if p.grad is not None:
                self.step_param(p)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            p.grad = None

    def state_dict(self) -> dict:
        return {i: self.optims[p].state_dict() for i, p in enumerate(self.params)}

    def load_state_dict(self, state: dict) -> None:
        for i, p in enumerate(self.params):
            self.optims[p].load_state_dict(state[i])
