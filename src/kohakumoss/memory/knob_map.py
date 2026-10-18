"""Top-level memory and precision knobs <-> MemoryPlan fields. With
MEMORY_PROFILE='manual' the memory knobs set the plan; with a named profile they must be
unset. Precision knobs are independent of the profile. See docs/concepts/profiles.md."""

MANUAL = "manual"
MEMORY_KNOBS = {
    "CHECKPOINT": "checkpoint",
    "STREAM": "stream",
    "STREAM_RESIDENT": "stream_resident",
    "PINNED": "pinned",
    "ACT_OFFLOAD": "act_offload",
    "OPTIM_STATE": "optim_state",
    "GRAD_OFFLOAD": "grad_offload",
    "FUSED_STEP": "fused_step",
}
PRECISION_KNOBS = {
    "BASE_PRECISION": "base_precision",
    "FP8_RECIPE": "fp8_recipe",
    "FP8_DGRAD": "fp8_dgrad",
}
PLACEMENT_KNOBS = ("STREAM", "STREAM_RESIDENT", "PINNED")


def knob_fields(k: dict, knobs: dict[str, str]) -> dict:
    """The plan fields of every knob in `knobs` that is set (not None)."""
    return {field: k[name] for name, field in knobs.items() if k.get(name) is not None}


def plan_fields(k: dict, profile: str) -> dict:
    """Plan fields from the top-level knobs: the memory knobs in manual mode, the
    precision knobs always."""
    fields = knob_fields(k, PRECISION_KNOBS)
    if profile == MANUAL:
        fields.update(knob_fields(k, MEMORY_KNOBS))
    return fields


def mixing_problems(k: dict, profile: str, profile_knob: str, knobs=MEMORY_KNOBS):
    """A named profile together with set memory knobs is an error naming both."""
    if profile in (MANUAL, "", None):
        return []
    named = [name for name in knobs if k.get(name) is not None]
    if not named:
        return []
    return [
        f"{profile_knob}={profile!r} sets every memory knob, so {', '.join(named)} must "
        f"stay None: drop them, or set {profile_knob}='manual' and set every memory "
        "knob yourself"
    ]
