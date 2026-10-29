"""HealConfig: one heal run (models, shrink, anneal, trainables, loss, schedule, eval),
built from the heal script's knobs and checked in one pass."""

from dataclasses import asdict, dataclass, field, fields

MODES = ("full_writers", "lokr")
LOKR_TARGETS = {
    "writers": r"layers\.\d+\.proj\.(o|ffn_down)$",
    "readers": r"layers\.\d+\.proj\.(q|k|v|ffn_up_gate)$",
    "all": r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)$",
}
WRITERS_FULL = r"layers\.\d+\.proj\.(o|ffn_down)\.weight$"
NORM_TARGETS = (
    r"layers\.\d+\.(norm1|norm2)$",
    r"text_embedder\.blocks\.\d+\.(norm1|norm2)$",
    r"final_norm_video$",
)


@dataclass(frozen=True)
class HealConfig:
    """Teacher: the original DiT; student: STUDENT (the teacher's file when empty),
    shrunk by REPORT / TAU when REPORT is set. MODE full_writers trains o / ffn_down in
    full and LoKr on the readers; lokr trains LoKr on LOKR_TARGET. Norm gains always
    train (TRAIN_NORMS deltas)."""

    teacher: str
    records: str
    out: str
    student: str = ""
    report: str = ""
    tau: float = 16.0
    anneal_steps: int = 0
    anneal_rbar: bool = True
    mode: str = "full_writers"
    lokr_target: str = "readers"
    lokr_factor: int = 8
    lr_norm: float = 2e-3
    lr_lokr: float = 5e-4
    lr_full: float = 3e-5
    hard_weight: float = 0.0
    conc_weight: float = 0.0
    conc_k: int = 8
    steps: int = 5000
    warmup: int = 100
    micro: int = 1
    grad_acc: int = 2
    text_drop: float = 0.1
    max_text_tokens: int = 1280
    held: int = 32
    eval_every: int = 500
    save_every: int = 1000
    grad_map: bool = True
    seed: int = 0
    teacher_memory: dict = field(default_factory=dict)
    student_memory: dict = field(default_factory=dict)

    @property
    def student_file(self) -> str:
        return self.student or self.teacher

    @property
    def lokr_regex(self) -> str:
        return LOKR_TARGETS[
            "readers" if self.mode == "full_writers" else self.lokr_target
        ]

    def record(self) -> dict:
        return asdict(self)


def from_knobs(g: dict) -> HealConfig:
    """HealConfig from UPPER_CASE knobs named after its fields."""
    return HealConfig(
        **{f.name: g[f.name.upper()] for f in fields(HealConfig) if f.name.upper() in g}
    )


def problems(c: HealConfig, paths: bool = True) -> list[str]:
    """Every problem of `c`; `paths=False` skips the required-path checks (a shipped
    recipe leaves the paths to the user)."""
    p = []
    for name in ("teacher", "records", "out") if paths else ():
        if not getattr(c, name):
            p.append(f"{name.upper()} is required")
    if c.mode not in MODES:
        p.append(f"MODE={c.mode!r} not in {MODES}")
    if c.mode == "lokr" and c.lokr_target not in LOKR_TARGETS:
        p.append(f"LOKR_TARGET={c.lokr_target!r} not in {sorted(LOKR_TARGETS)}")
    if paths and c.anneal_steps > 0 and not c.report:
        p.append(
            "ANNEAL_STEPS > 0 anneals the shrink of a calibration REPORT: set REPORT"
        )
    if c.anneal_steps > c.steps:
        p.append(f"ANNEAL_STEPS={c.anneal_steps} exceeds STEPS={c.steps}")
    for name in ("lr_norm", "lr_lokr", "lr_full", "tau"):
        if not getattr(c, name) > 0:
            p.append(f"{name.upper()} must be > 0")
    for name in ("hard_weight", "conc_weight", "text_drop"):
        if getattr(c, name) < 0:
            p.append(f"{name.upper()} must be >= 0")
    for name in ("steps", "micro", "grad_acc", "conc_k", "lokr_factor", "eval_every"):
        if getattr(c, name) < 1:
            p.append(f"{name.upper()} must be >= 1")
    if c.warmup < 0 or c.held < 1 or c.save_every < 0:
        p.append("WARMUP >= 0, HELD >= 1 and SAVE_EVERY >= 0 are required")
    return p
