"""Cell formatters of the bench report tables (numbers, GiB, VRAM phases, non-finite
steps)."""

GIB_MIB = 1024


def _num(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _gib(mib) -> float | None:
    return None if mib is None else mib / GIB_MIB


def _phase_vram(record: dict, phase: str) -> tuple:
    """(nvidia-smi peak GiB, steady GiB) of one phase ("prewarm" | "step") of a run:
    the process column, else (WDDM) the card above its idle baseline."""
    smi = record.get("vram_smi_phases") or {}
    proc = smi.get(f"{phase}_smi_process_peak_mib")
    if proc is not None:
        return _gib(proc), _gib(smi.get(f"{phase}_smi_process_steady_mib"))
    return (
        _gib(smi.get(f"{phase}_smi_card_above_idle_mib")),
        _gib(smi.get(f"{phase}_smi_card_steady_above_idle_mib")),
    )


def _skips(s: dict) -> str:
    """Non-finite steps with their step and the loss scale after it ("1 @ 4 (32768)")."""
    at = s.get("nonfinite_at") or []
    if not at:
        return str(s.get("nonfinite_steps", "-"))
    where = ", ".join(
        f"{step} ({'-' if scale is None else f'{scale:g}'})" for step, scale in at[:4]
    )
    return f"{len(at)} @ {where}"
