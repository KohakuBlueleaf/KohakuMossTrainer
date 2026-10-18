"""AdapterConfig: what to train and how, resolved from the train-script knobs."""

import inspect
from dataclasses import dataclass, field

from lycoris.wrapper import LycorisNetwork

from kohakumoss.adapters.kinds import BY_ALGO, lycoris_algo
from kohakumoss.adapters.train.tlora import MASK_KEYS

MODES = ("bypass", "merge")
BACKENDS = ("auto", "triton", "tilelang", "compile", "torch")
COMPILE_DYNAMIC = ("tokens", "static", "dynamic")
RESERVED = {
    "lora_dim",
    "alpha",
    "dropout",
    "rank_dropout",
    "module_dropout",
    "multiplier",
    "network_module",
}


def _parameters(fn) -> set[str]:
    kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    sig = inspect.signature(fn)
    return {n for n, p in sig.parameters.items() if p.kind not in kinds} - {"self"}


def accepted_kwargs(algo: str) -> set[str]:
    """ALGO_KWARGS names the LyCORIS module (or network) of `algo` takes."""
    module = BY_ALGO[algo].module
    return (_parameters(module.__init__) | _parameters(LycorisNetwork.__init__)) - {
        "lora_name",
        "org_module",
        "module",
    }


@dataclass(frozen=True)
class AdapterConfig:
    """One adapter configuration; `algo_kwargs` go to the LyCORIS module constructor
    (T-LoRA's `mask_*` keys configure its timestep mask instead). `backend` "auto"
    autotunes each kernel op against its compiled body; `compile_dynamic` is the compile
    candidate's shape mode; `fuse_base_add` puts the base add inside the op."""

    algo: str = "lora"
    preset: str | dict = "attn-mlp"
    dim: int = 32
    alpha: float = 32.0
    algo_kwargs: dict = field(default_factory=dict)
    dora: bool = False
    dora_on_output: bool = True
    dora_detach: bool = False
    dropout: float = 0.0
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    mode: str = "bypass"
    train_norms: bool = False
    multiplier: float = 1.0
    prefix: str = "lora_unet"
    backend: str = "auto"
    compile_dynamic: str = "tokens"
    lr_ratio_up: float = 1.0
    seed: int = 0
    fuse_base_add: bool = True

    @property
    def lycoris_algo(self) -> str:
        return lycoris_algo(self.algo)

    @property
    def record(self):
        return BY_ALGO[self.algo]

    @property
    def has_step_state(self) -> bool:
        """The adapters carry per-step state: T-LoRA masks, DyLoRA ranks or
        module-dropout draws."""
        rec = BY_ALGO.get(self.algo)
        drawn = rec is not None and (rec.per_token_mask or rec.draw_step is not None)
        return drawn or self.module_dropout > 0

    def mask_kwargs(self) -> dict:
        """T-LoRA mask settings (min_rank / alpha / max_timestep) from ALGO_KWARGS."""
        return {MASK_KEYS[k]: v for k, v in self.algo_kwargs.items() if k in MASK_KEYS}

    def validate(self) -> list[str]:
        """Every problem with this configuration (empty when valid)."""
        problems = []
        rec = BY_ALGO.get(self.algo)
        if rec is None:
            problems.append(f"ALGO={self.algo!r} not in {sorted(BY_ALGO)}")
        if self.mode not in MODES:
            problems.append(f"ADAPTER_MODE={self.mode!r} not in {MODES}")
        if self.backend not in BACKENDS:
            problems.append(f"adapter backend {self.backend!r} not in {BACKENDS}")
        if self.compile_dynamic not in COMPILE_DYNAMIC:
            problems.append(
                f"compile_dynamic={self.compile_dynamic!r} not in {COMPILE_DYNAMIC}"
            )
        if self.dim <= 0 and not (rec is not None and rec.rankless):
            problems.append(f"DIM must be positive, got {self.dim}")
        if self.dora and self.dropout > 0:
            problems.append("DORA ignores input dropout; set DROPOUT['input']=0")
        for name in ("dropout", "rank_dropout", "module_dropout"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                problems.append(f"{name}={value} outside [0, 1)")
        if rec is not None:
            problems += self._kind_problems(rec)
        return problems

    def _kind_problems(self, rec) -> list[str]:
        out = []
        if self.dora and not rec.supports_dora:
            out.append(f"DORA does not apply to ALGO={self.algo!r}")
        if rec.per_token_mask and self.mode == "merge":
            out.append(
                "ALGO='tlora' masks ranks per token, which no merged weight expresses; "
                "use ADAPTER_MODE='bypass'"
            )
        return out + self._kwarg_problems()

    def _kwarg_problems(self) -> list[str]:
        own = set(MASK_KEYS) if self.record.per_token_mask else set()
        accepted = accepted_kwargs(self.algo)
        out = []
        for key in self.algo_kwargs:
            if key in own:
                continue
            if key in RESERVED:
                out.append(f"ALGO_KWARGS[{key!r}] is set by its own knob")
            elif key not in accepted:
                out.append(f"ALGO_KWARGS[{key!r}] is not a {self.algo} module argument")
        return out

    def module_kwargs(self) -> dict:
        """Constructor kwargs shared by every LyCORIS module of this configuration."""
        kwargs = {k: v for k, v in self.algo_kwargs.items() if k not in MASK_KEYS}
        kwargs.setdefault("bypass_mode", self.mode == "bypass")
        if self.dora:
            kwargs["weight_decompose"] = True
            kwargs["wd_on_out"] = self.dora_on_output
        return kwargs


def config_from_knobs(cfg: dict) -> AdapterConfig:
    """AdapterConfig from the train-script globals (ALGO, PRESET, DIM, ...)."""
    dropout = cfg.get("DROPOUT", {}) or {}
    preset = cfg.get("PRESET", "attn-mlp")
    if cfg.get("TRAIN_TEXT_REFINER", False) and isinstance(preset, str):
        if "+text-refiner" not in preset and not preset.endswith(".toml"):
            preset = f"{preset}+text-refiner"
    algo_kwargs = dict(cfg.get("ALGO_KWARGS", {}) or {})
    return AdapterConfig(
        algo=cfg.get("ALGO", "lora"),
        preset=preset,
        dim=int(cfg.get("DIM", 32)),
        alpha=float(cfg.get("ALPHA", 32.0)),
        algo_kwargs=algo_kwargs,
        dora=bool(cfg.get("DORA", False)),
        dora_on_output=bool(algo_kwargs.pop("wd_on_out", True)),
        dora_detach=bool(cfg.get("DORA_DETACH", False)),
        dropout=float(dropout.get("input", 0.0)),
        rank_dropout=float(dropout.get("rank", 0.0)),
        module_dropout=float(dropout.get("module", 0.0)),
        mode=cfg.get("ADAPTER_MODE", "bypass"),
        train_norms=bool(cfg.get("TRAIN_NORMS", False)),
        backend=(cfg.get("KERNELS", {}) or {}).get("lycoris", "auto"),
        lr_ratio_up=float(cfg.get("LORAPLUS_RATIO", 1.0)),
        seed=int(cfg.get("SEED", 0)),
    )


def inference_config(fields: dict, multiplier: float) -> AdapterConfig:
    """A saved configuration rebuilt for inference / export on any device: no dropout,
    the device's own kernel backend (a training-time backend pin does not carry over).
    """
    known = set(AdapterConfig.__dataclass_fields__)
    kept = {k: v for k, v in fields.items() if k in known}
    quiet = {"dropout": 0.0, "rank_dropout": 0.0, "module_dropout": 0.0}
    return AdapterConfig(
        **{**kept, **quiet, "backend": "auto", "multiplier": multiplier}
    )
