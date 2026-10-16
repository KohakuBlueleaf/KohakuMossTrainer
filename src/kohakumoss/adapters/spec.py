"""AdapterSpec: what to train and how, resolved from the train-script knobs."""

from dataclasses import dataclass, field

ALGO_ALIASES = {
    "lora": "lora",
    "locon": "locon",
    "loha": "loha",
    "lokr": "lokr",
    "oft": "diag-oft",
    "diag-oft": "diag-oft",
    "boft": "boft",
    "glora": "glora",
    "ia3": "ia3",
    "dylora": "dylora",
    "tlora": "tlora",
    "full": "full",
}
MODES = ("bypass", "merge")
BACKENDS = ("auto", "triton", "tilelang", "compile", "torch")
COMPILE_DYNAMIC = ("tokens", "static", "dynamic")


@dataclass(frozen=True)
class AdapterSpec:
    """One adapter configuration; `algo_kwargs` go to the LyCORIS module constructor.

    `preset` is a MOSS preset expression ("attn-mlp+text-refiner"), a LyCORIS-style
    toml path, or a preset dict. `backend` pins the LyCORIS kernel tier.
    """

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

    @property
    def lycoris_algo(self) -> str:
        return ALGO_ALIASES[self.algo]

    def validate(self) -> list[str]:
        """Every problem with this spec (empty when valid)."""
        problems = []
        if self.algo not in ALGO_ALIASES:
            problems.append(f"ALGO={self.algo!r} not in {sorted(ALGO_ALIASES)}")
        if self.mode not in MODES:
            problems.append(f"ADAPTER_MODE={self.mode!r} not in {MODES}")
        if self.backend not in BACKENDS:
            problems.append(f"adapter backend {self.backend!r} not in {BACKENDS}")
        if self.compile_dynamic not in COMPILE_DYNAMIC:
            problems.append(
                f"LYCORIS_COMPILE_DYNAMIC={self.compile_dynamic!r} not in {COMPILE_DYNAMIC}"
            )
        if self.dim <= 0 and self.algo not in ("full", "ia3"):
            problems.append(f"DIM must be positive, got {self.dim}")
        if self.dora and self.algo not in ("lora", "locon", "loha", "lokr"):
            problems.append(f"DORA applies to lora/locon/loha/lokr, not {self.algo}")
        if self.algo == "full" and self.mode == "bypass":
            problems.append(
                "ALGO='full' trains the weight itself; use ADAPTER_MODE='merge'"
            )
        for name in ("dropout", "rank_dropout", "module_dropout"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                problems.append(f"{name}={value} outside [0, 1)")
        return problems

    def module_kwargs(self) -> dict:
        """Constructor kwargs shared by every LyCORIS module of this spec."""
        kwargs = dict(self.algo_kwargs)
        kwargs.setdefault("bypass_mode", self.mode == "bypass" and self.algo != "full")
        if self.dora:
            kwargs["weight_decompose"] = True
            kwargs["wd_on_out"] = self.dora_on_output
        return kwargs


def spec_from_knobs(cfg: dict) -> AdapterSpec:
    """AdapterSpec from the train-script globals (ALGO, PRESET, DIM, ...)."""
    dropout = cfg.get("DROPOUT", {}) or {}
    preset = cfg.get("PRESET", "attn-mlp")
    if cfg.get("TRAIN_TEXT_REFINER", False) and isinstance(preset, str):
        if "+text-refiner" not in preset and not preset.endswith(".toml"):
            preset = f"{preset}+text-refiner"
    algo_kwargs = dict(cfg.get("ALGO_KWARGS", {}) or {})
    return AdapterSpec(
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
        compile_dynamic=cfg.get("LYCORIS_COMPILE_DYNAMIC", "tokens"),
        lr_ratio_up=float(cfg.get("LORAPLUS_RATIO", 1.0)),
    )
