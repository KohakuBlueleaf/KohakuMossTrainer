"""Optimiser parameter groups of an adapter set: weight-decay rules, Muon eligibility
and the LoRA+ ratio on each kind's zero-initialised factors."""

NO_DECAY = ("dora_scale", "rescale", "scalar", "lambda_layer", "alpha")
WD_RULES = ("moss", "none", "all")


def is_up(name: str, up_side: tuple[str, ...]) -> bool:
    """`name` (relative to its LyCORIS module) is one of the kind's LoRA+ factors."""
    return any(name == tag or name.startswith(tag + ".") for tag in up_side)


def _no_decay_name(name: str, p) -> bool:
    return p.dim() < 2 or any(tag in name for tag in NO_DECAY)


def _decay(rules: str, name: str, p, weight_decay: float) -> float:
    match rules:
        case "moss":
            return 0.0 if _no_decay_name(name, p) else weight_decay
        case "none":
            return 0.0
        case "all":
            return weight_decay
    raise ValueError(f"unknown WD_RULES {rules!r}; known {WD_RULES}")


def _muon(use_muon: bool, name: str, p) -> bool:
    """2-D factors with both sides > 1 (DoRA magnitudes and scalars stay on AdamW)."""
    return (
        use_muon and p.dim() == 2 and min(p.shape) > 1 and not _no_decay_name(name, p)
    )


def build(
    entries,
    lr: float,
    weight_decay: float,
    rules: str,
    use_muon: bool,
    ratio: float,
) -> list[dict]:
    """`entries`: (full name, name in its module, param, up_side) per trainable param."""
    buckets: dict[tuple, dict] = {}
    for full, name, p, up_side in entries:
        wd = _decay(rules, name, p, weight_decay)
        up = ratio != 1.0 and is_up(name, up_side)
        muon = _muon(use_muon, name, p)
        group = buckets.setdefault(
            (wd, up, muon),
            {
                "params": [],
                "names": [],
                "splits": [],
                "lr": lr * (ratio if up else 1.0),
                "weight_decay": wd,
                "decay_to_one": False,
                "decay_to_one_coef": 0.0,
                "use_muon": muon,
            },
        )
        group["params"].append(p)
        group["names"].append(full)
        group["splits"].append(1)
    return [buckets[k] for k in sorted(buckets)]
