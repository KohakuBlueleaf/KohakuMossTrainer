"""The release and encoder-placement tables of the bench report.
See docs/performance/bench-matrix.md#release."""

from kohakumoss_bench.report.cells import _num, _phase_vram, _skips
from kohakumoss_bench.report.results import phase_peaks
from kohakumoss_common.markdown import md_table

RELEASE_COLUMNS = (
    "run",
    "card",
    "status",
    "samples/s",
    "tok/s",
    "te / vae s/step",
    "step peak GiB",
    "smi prewarm / step peak GiB",
    "torch prewarm alloc / reserved",
    "torch step alloc / reserved",
    "prewarm > step",
    "reserved first -> last",
    "past budget",
    "loss scale",
    "non-finite",
)


PLACEMENT_COLUMNS = (
    "run",
    "card",
    "text",
    "VAE",
    "auto picked text / VAE",
    "status",
    "samples/s",
    "te s/step (share)",
    "vae s/step (share)",
    "te H2D GB/step",
    "step peak GiB",
    "smi prewarm peak / steady GiB",
    "smi step peak / steady GiB",
)


def _pair(a, b, digits: int = 2) -> str:
    return f"{_num(a, digits)} / {_num(b, digits)}"


def _yes(value: bool | None) -> str:
    return "-" if value is None else ("yes" if value else "no")


def release_rows(records: list[dict]) -> list[str]:
    """The release rows: throughput, encoder seconds per step (means), the steady step
    peak, the prewarm phase against the step phase (nvidia-smi peak, torch max
    allocated / reserved, GiB; `results.phase_peaks`) and whether the prewarm peaked
    above the steps, the fp16 loss scale, non-finite steps."""
    rows = []
    for r in records:
        if r.get("kind") != "release":
            continue
        s, p = r.get("summary", {}), phase_peaks(r)
        cap = r.get("cap_gib") or 0
        budget = (
            "-" if p["past_budget"] is None else f"{_yes(p['past_budget'])} ({cap:g})"
        )
        rows.append(
            (
                r["run"],
                r.get("card", "-"),
                r.get("status"),
                _num(s.get("samples_per_s_median")),
                _num(s.get("tokens_per_s_median"), 0),
                _pair(s.get("te_time_mean"), s.get("vae_time_mean"), 3),
                _num(s.get("vram_step_peak_median")),
                _pair(p["smi_prewarm"], p["smi_step"]),
                _pair(p["torch_prewarm_alloc"], p["torch_prewarm_reserved"]),
                _pair(p["torch_step_alloc"], p["torch_step_reserved"]),
                _yes(p["over"]),
                f"{_num(p['reserved_first'])} -> {_num(p['reserved_last'])}",
                budget,
                _num(s.get("loss_scale_median"), 0),
                _skips(s),
            )
        )
    return md_table(RELEASE_COLUMNS, rows)


def _setup(record: dict) -> str:
    """A placement run's setup: its name up to the text placement."""
    return record["run"].rsplit("-te", 1)[0]


def placement_rows(records: list[dict]) -> list[str]:
    """Encoder placements under no cache: per setup, each text / VAE placement with
    what `auto` picked for the same setup on this pool (its log's placement lines),
    throughput, encoder seconds per step and share (means), text-encoder H2D GB per
    step, the steady step peak and nvidia-smi peak / steady per phase."""
    rows = [r for r in records if r.get("kind") == "placement"]
    auto = {
        _setup(r): r.get("summary", {})
        for r in rows
        if r.get("tags", {}).get("text_memory") == "auto"
    }
    out = []
    for r in rows:
        t, s = r.get("tags", {}), r.get("summary", {})
        a = auto.get(_setup(r), {})
        pre, step = _phase_vram(r, "prewarm"), _phase_vram(r, "step")
        out.append(
            (
                r["run"],
                r.get("card", "-"),
                t.get("text_memory", "-"),
                t.get("vae_memory", "-"),
                f"{a.get('te_placement', '-')} / {a.get('vae_placement', '-')}",
                r.get("status"),
                _num(s.get("samples_per_s_median")),
                f"{_num(s.get('te_time_mean'), 3)} ({_num(s.get('te_share_mean'))})",
                f"{_num(s.get('vae_time_mean'), 3)} ({_num(s.get('vae_share_mean'))})",
                _num(s.get("te_h2d_gb_mean")),
                _num(s.get("vram_step_peak_median")),
                _pair(*pre),
                _pair(*step),
            )
        )
    return md_table(PLACEMENT_COLUMNS, out)
