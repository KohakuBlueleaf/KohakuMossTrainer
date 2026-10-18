"""The profile catalog as a table: each profile's placement and its estimated peak (GiB)
for the MOSS base DiT at 1 MP, LoRA and full FT. See docs/concepts/profiles.md."""

from kohakumoss.memory.estimate import GIB, Workload, estimate
from kohakumoss.memory.profiles import CATALOG, ORDER, PROFILES, resolve
from kohakumoss.types.arch import PRESETS, MossArchConfig
from kohakumoss_common.markdown import md_table

IMG_TOKENS = 4096
TXT_TOKENS = 300
LORA_RANK = 16
HEADER = (
    "profile",
    "checkpoint",
    "weights",
    "act_offload",
    "optimizer",
    "grad_offload",
    "LoRA GiB",
    "full FT GiB",
)


def lora_params(arch: MossArchConfig, rank: int = LORA_RANK) -> int:
    """LoRA parameters of rank `rank` on every block linear (q, k, v, o, the FFN)."""
    d, f = arch.dim, arch.ffn_dim
    return arch.depth * rank * (4 * 2 * d + 3 * (d + f))


def workloads(arch: MossArchConfig = PRESETS["base"]) -> tuple[Workload, Workload]:
    """(LoRA, full FT) micro-steps at 1 MP: 4096 image tokens, 300 text tokens, bf16."""
    tokens = dict(img_tokens=IMG_TOKENS, txt_tokens=TXT_TOKENS)
    return (
        Workload(arch, trainable_params=lora_params(arch), **tokens),
        Workload(arch, full_ft=True, **tokens),
    )


def _peak(work: Workload, name: str) -> str:
    try:
        plan = resolve(name, device="cpu", trainable_base=work.full_ft)
    except ValueError as err:
        return "n/a: " + str(err).split(": ", 1)[1]
    return f"{estimate(work, plan).total / GIB:.1f}"


def rows(
    arch: MossArchConfig = PRESETS["base"], names: tuple[str, ...] | None = None
) -> list[tuple[str, ...]]:
    """One row per profile of `names` (default: each short name, then each catalog
    entry), in HEADER order."""
    lora, full = workloads(arch)
    out = []
    for name in (*ORDER, *CATALOG) if names is None else names:
        p = PROFILES[name]
        stream = p.get("stream", "off")
        weights = "resident" if stream == "off" else f"ring {p['stream_resident']}"
        fused = " fused" if p.get("fused_step") else ""
        out.append(
            (
                f"`{name}`",
                p.get("checkpoint", "block"),
                weights,
                "on" if p.get("act_offload") else "off",
                p.get("optim_state", "gpu") + fused,
                "on" if p.get("grad_offload") else "off",
                _peak(lora, name),
                _peak(full, name),
            )
        )
    return out


def markdown(
    arch: MossArchConfig = PRESETS["base"], names: tuple[str, ...] | None = None
) -> str:
    """The table of `names` (default: every profile) as markdown lines (header, rule,
    rows)."""
    return "\n".join(md_table(HEADER, rows(arch, names)))
