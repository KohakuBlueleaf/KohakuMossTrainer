"""Model-card assets of the fp16-fix study: charts (svg + webp on an opaque white
background, readable on light and dark pages) and labelled image strips (webp; every
comparison in four columns: fp16 without the fix | fp16-fixed | bf16 | fp32) from the
outputs of fp16_study.py, fp16_quality.py, the ComfyUI runner and
scripts/export/fp16_fix.py, plus one merged report per variant.

    kogine run scripts/probe/fp16_charts.py --set RUNS=<fp16 runs dir> --set OUT=<staging>

RUNS holds study/<variant>/, quality/<variant>/, comfy/<variant>/ and
export/<variant>_export.json (fp16_fix.py REPORT).
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from kohakumoss.probe.card_panels import INK, PAPER, strip
from kohakumoss_common.knobs import check_script

RUNS = ""  # fp16 runs dir (see the module docstring)
OUT = ""  # staging dir: images/, reports/
VARIANTS = ("base", "sft")
FP16_MAX = 65504.0
COLUMNS = (  # the compared configs left of the fp32 reference, with their labels
    ("official_fp16", "fp16, no fix"),
    ("fix_fp16", "fp16-fixed"),
    ("bf16", "bf16"),
)
COLORS = {
    "orig": "#e4572e",
    "sampled": "#f3a712",
    "fixed": "#2e86ab",
    "folded": "#29bf12",
    "o": "#a663cc",
    "ffn_down": "#2e86ab",
    "ffn_up_gate": "#29bf12",
    "logit": "#f3a712",
    "text": "#e4572e",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "savefig.transparent": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": INK,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "font.size": 11,
            "figure.constrained_layout.use": True,
        }
    )


def save(fig, out: Path, name: str) -> list[str]:
    """`name`.svg and `name`.webp (rendered at 2x) under out/images."""
    d = out / "images"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.svg", bbox_inches="tight")
    png = d / f"{name}.tmp.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    Image.open(png).convert("RGB").save(
        d / f"{name}.webp", "WEBP", quality=92, method=6
    )
    png.unlink()
    plt.close(fig)
    return [f"images/{name}.svg", f"images/{name}.webp"]


def fp16_line(ax) -> None:
    ax.axhline(FP16_MAX, color="#d62828", ls="--", lw=1.2)
    ax.text(0.5, FP16_MAX * 1.25, "fp16 max 65504", color="#d62828", fontsize=9)


def stream_chart(variant: str, study: dict, out: Path) -> list[str]:
    """Per-block max |x| of the joint stream (probe grid, sampling trajectories, the
    fixed fp16 run) and of the block's other values, log scale, fp16 limit marked."""
    probe = study["probe"]
    c = probe["chosen_c"]
    row = next(r for r in study["sweep"] if r["c"] == c)
    pb, ps = probe["per_block"], probe["per_block_sampled"]
    x = np.arange(len(pb["stream"]))
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    a.semilogy(
        x,
        pb["stream"],
        "o-",
        ms=3,
        color=COLORS["orig"],
        label="official, probe grid (fp32)",
    )
    a.semilogy(
        x,
        ps["stream"],
        "s-",
        ms=3,
        color=COLORS["sampled"],
        label="official, real sampling (fp32)",
    )
    a.semilogy(
        x,
        study["plain"]["fp16_per_block"]["stream"],
        "d-",
        ms=3,
        color=COLORS["o"],
        label="official, plain fp16 run",
    )
    a.semilogy(
        x,
        row["per_block"]["stream"],
        "^-",
        ms=3,
        color=COLORS["fixed"],
        label=f"fp16-fixed (c = 1/{round(1 / c)}), fp16 run",
    )
    peak = int(np.argmax(pb["stream"]))
    a.annotate(
        f"block {peak}: {pb['stream'][peak]:,.0f} = {pb['stream'][peak] / FP16_MAX:.0%} of fp16 max",
        (peak, pb["stream"][peak]),
        xytext=(max(peak - 22, 0), pb["stream"][peak] / 6),
        color=COLORS["orig"],
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": COLORS["orig"]},
    )
    fp16_line(a)
    a.set_ylabel("max |x|, joint residual stream")
    a.set_title(
        f"MOSS-Image-{variant.upper() if variant == 'sft' else variant.title()}: residual stream per block"
    )
    a.grid(True, which="both")
    a.legend(loc="center right", fontsize=9)
    for site in ("o", "ffn_down", "ffn_up_gate", "logit"):
        if site in pb:
            b.semilogy(
                x, pb[site], "o-", ms=3, color=COLORS[site], label=f"{site} (fp32)"
            )
    fp16_line(b)
    b.set_ylabel("max |x|")
    b.set_xlabel("joint block")
    b.grid(True, which="both")
    b.legend(loc="lower right", fontsize=9, ncol=2)
    return save(fig, out, f"{variant}_stream_per_block")


def plain_chart(variant: str, study: dict, out: Path) -> list[str]:
    """Plain fp16 on the official weights: share of non-finite outputs per resolution x
    sigma (over the prompts), headroom 65504 / max |stream| per block (official fp16,
    official bf16, fp16-fixed), and the first block whose output is non-finite when any
    forward overflowed."""
    plain = study["plain"]
    rows = plain["fp16_forward_rows"]
    res = [tuple(r) for r in study["resolutions"]]
    sig = study["sigmas"]
    grid = np.zeros((len(res), len(sig)))
    for i, r in enumerate(res):
        for j, s in enumerate(sig):
            cell = [
                q["nonfinite"] for q in rows if tuple(q["res"]) == r and q["sigma"] == s
            ]
            grid[i, j] = sum(cell) / max(len(cell), 1)
    first = [q["first_bad_block"] for q in rows if q["first_bad_block"] is not None]
    c = study["probe"]["chosen_c"]
    fixed = next(r for r in study["sweep"] if r["c"] == c)["per_block"]["stream"]
    ratios = [1.3, 1.6] + ([1] if first else [])
    fig, axes = plt.subplots(
        1,
        len(ratios),
        figsize=(6 * len(ratios), 3.8),
        gridspec_kw={"width_ratios": ratios},
    )
    a, h = axes[0], axes[1]
    im = a.imshow(grid, vmin=0, vmax=1, cmap="magma", aspect="auto")
    a.set_xticks(range(len(sig)), [str(s) for s in sig])
    a.set_yticks(range(len(res)), [f"{rh}x{rw}" for rh, rw in res])
    a.set_xlabel("sigma")
    a.set_ylabel("resolution")
    a.set_title(f"plain fp16: non-finite in {len(first)} of {len(rows)} forwards")
    for i in range(len(res)):
        for j in range(len(sig)):
            a.text(
                j,
                i,
                f"{grid[i, j]:.0%}",
                ha="center",
                va="center",
                color="#ffffff" if grid[i, j] < 0.6 else "#000000",
                fontsize=8,
            )
    fig.colorbar(im, ax=a, fraction=0.04)
    x = np.arange(len(fixed))
    for vals, mark, color, name in (
        (plain["fp16_per_block"]["stream"], "d-", COLORS["o"], "official, fp16"),
        (plain["bf16_per_block"]["stream"], "s-", COLORS["sampled"], "official, bf16"),
        (fixed, "^-", COLORS["fixed"], f"fp16-fixed (c = 1/{round(1 / c)}), fp16"),
    ):
        h.semilogy(x, [FP16_MAX / v for v in vals], mark, ms=3, color=color, label=name)
    h.axhline(1.0, color="#d62828", ls="--", lw=1.2)
    h.set_xlabel("joint block")
    h.set_ylabel("headroom = 65504 / max |stream|")
    h.set_title("margin to the fp16 limit per block")
    h.grid(True, which="both")
    h.legend(fontsize=8)
    if first:
        b = axes[2]
        b.hist(first, bins=np.arange(-0.5, len(fixed) + 0.5, 1), color=COLORS["orig"])
        b.set_xlabel("first joint block with a non-finite output")
        b.set_ylabel("forwards")
        b.grid(True)
    return save(fig, out, f"{variant}_plain_fp16")


def sweep_chart(variant: str, study: dict, out: Path) -> list[str]:
    """Error vs fp32 and headroom against c; weight / stream underflow; stock-eps error."""
    sweep = study["sweep"]
    cs = [r["c"] for r in sweep]
    xs = [math.log2(c) for c in cs]
    labels = [f"1/{round(1 / c)}" for c in cs]
    chosen = math.log2(study["probe"]["chosen_c"])
    n = len(sweep[0]["vs_fp32"])
    bf16 = study["plain"]["bf16_vs_fp32"][:n]
    bf16_lpips = sum(r["lpips"] for r in bf16) / len(bf16)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    a = axes[0]
    a.plot(
        xs,
        [r["mean_lpips"] for r in sweep],
        "o-",
        color=COLORS["fixed"],
        label="fp16-fixed vs fp32",
    )
    a.axhline(
        bf16_lpips, color=COLORS["sampled"], ls=":", label="official bf16 vs fp32"
    )
    a.set_ylabel(f"mean LPIPS vs fp32 ({n} images)")
    a.legend(fontsize=9)
    b = axes[1]
    b.semilogy(xs, [r["headroom"] for r in sweep], "o-", color=COLORS["folded"])
    b.axhline(1.0, color="#d62828", ls="--", lw=1)
    goal = FP16_MAX / study["target"]
    b.axhline(goal, color=COLORS["folded"], ls=":", lw=1)
    b.text(
        min(xs), goal * 1.15, f"target {goal:.0f}x", color=COLORS["folded"], fontsize=9
    )
    b.set_ylabel("headroom = 65504 / fp16 stream max")
    d = axes[2]
    tot = [r["weights"]["total"] for r in sweep]
    d.semilogy(
        xs,
        [max((t["flushed"] + t["subnormal"]) / t["n"], 1e-12) for t in tot],
        "o-",
        color=COLORS["o"],
        label="scaled weights: subnormal or flushed",
    )
    d.semilogy(
        xs,
        [max(r["stream"]["subnormal_frac"], 1e-12) for r in sweep],
        "s-",
        color=COLORS["ffn_up_gate"],
        label="stream values: subnormal",
    )
    eps = [max(q["stock_rel"] for q in study["eps"][str(c)]) for c in cs]
    d.semilogy(
        xs,
        [max(e, 1e-12) for e in eps],
        "^-",
        color=COLORS["text"],
        label="stock-eps rel. error (fp32)",
    )
    d.set_ylabel("fraction / relative error")
    d.legend(loc="lower left", fontsize=8)
    for ax in axes:
        ax.axvline(chosen, color=INK, ls="--", lw=1)
        ax.set_xticks(xs, labels)
        ax.set_xlabel("c")
        ax.grid(True, which="both")
    fig.suptitle(f"choosing c ({variant}): dashed line = chosen c")
    return save(fig, out, f"{variant}_c_sweep")


def lpips_tag(scores: dict, name: str, key: str) -> str:
    """ "  LPIPS* x" of image `key` of config `name` in a score dict, "" when absent."""
    rows = scores.get(name, {}).get("rows", [])
    v = next((r["lpips"] for r in rows if r["key"] == key), None)
    return "" if v is None else f"  LPIPS* {v:.3f}"


def gallery(variant: str, quality: dict, runs: Path, out: Path) -> list[dict]:
    """COLUMNS | fp32 per runtime, prompt and seed; the same four with each adapter
    (first seed), fp32 + adapter as the reference."""
    made = []
    dirs = {
        "kohakumoss": runs / "quality" / variant / "images" / "kohakumoss",
        "diffusers": runs / "quality" / variant / "images" / "diffusers",
        "comfyui": runs / "comfy" / variant,
    }
    for rt, d in dirs.items():
        q = quality.get(rt, {})
        for p in range(len(quality["prompts"])):
            for s in quality["seeds"]:
                key = f"p{p}_s{s}"
                panels = [
                    (
                        d / f"{name}_{key}.png",
                        text + lpips_tag(q.get("vs_fp32", {}), name, key),
                    )
                    for name, text in COLUMNS
                ]
                panels.append((d / f"fp32_{key}.png", "fp32 (* = LPIPS to it)"))
                dest = out / "images" / "gallery" / f"{variant}_{rt}_{key}.webp"
                if strip(panels, dest):
                    made.append(
                        {"runtime": rt, "key": key, "file": str(dest.relative_to(out))}
                    )
        for i in range(4):
            key = f"p0_s{quality['seeds'][0]}"
            if not (d / f"fp32_lora{i}_{key}.png").is_file():
                break
            scores = q.get("lora", {}).get(f"lora{i}", {}).get("vs_fp32", {})
            for p in range(len(quality["prompts"])):
                key = f"p{p}_s{quality['seeds'][0]}"
                panels = [
                    (
                        d / f"{name}_lora{i}_{key}.png",
                        f"{text} + LoRA" + lpips_tag(scores, f"{name}_lora{i}", key),
                    )
                    for name, text in COLUMNS
                ]
                panels.append((d / f"fp32_lora{i}_{key}.png", "fp32 + LoRA (*)"))
                dest = out / "images" / "gallery" / f"{variant}_{rt}_lora{i}_{key}.webp"
                if strip(panels, dest):
                    made.append(
                        {
                            "runtime": rt,
                            "key": key,
                            "lora": i,
                            "file": str(dest.relative_to(out)),
                        }
                    )
    return made


def failure(variant: str, runs: Path, out: Path) -> list[str]:
    """The study's plain fp16 / fixed fp16 / bf16 / fp32 on prompts 0 and 1."""
    d = runs / "study" / variant / "images"
    c = json.loads((runs / "study" / variant / "report.json").read_text())["probe"][
        "chosen_c"
    ]
    made = []
    for p in range(2):
        dest = out / "images" / f"{variant}_plain_fp16_p{p}.webp"
        if strip(
            [
                (d / f"official_fp16_p{p}.png", "fp16, no fix"),
                (d / f"fix_c{round(1 / c)}_p{p}.png", "fp16-fixed"),
                (d / f"official_bf16_p{p}.png", "bf16"),
                (d / f"fp32_p{p}.png", "fp32"),
            ],
            dest,
        ):
            made.append(str(dest.relative_to(out)))
    return made


def sweep_strip(variant: str, study: dict, runs: Path, out: Path) -> list[str]:
    """Prompt 0 at every swept c next to fp32."""
    d = runs / "study" / variant / "images"
    made = []
    panels = [
        (
            d / f"fix_c{round(1 / r['c'])}_p0.png",
            f"c = 1/{round(1 / r['c'])}  LPIPS {r['vs_fp32'][0]['lpips']:.3f}",
        )
        for r in study["sweep"]
    ]
    panels.append((d / "fp32_p0.png", "fp32 reference"))
    for i in range(0, len(panels), 4):
        dest = out / "images" / f"{variant}_c_sweep_p0_{i // 4}.webp"
        if strip(panels[i : i + 4], dest):
            made.append(str(dest.relative_to(out)))
    return made


def md_table(head: list[str], rows: list[list]) -> list[str]:
    """Markdown table lines."""
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return lines + [""]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def score_row(name: str, v: dict) -> list:
    """[name, images, mean LPIPS, mean SSIM, mean PSNR] of one fp16_quality.py score entry."""
    ssim = f"{v['mean_ssim']:.4f}" if "mean_ssim" in v else ""
    return [
        name,
        len(v["rows"]),
        f"{v['mean_lpips']:.4f}",
        ssim,
        f"{v['mean_psnr']:.2f}",
    ]


def tables(variant: str, study: dict, quality: dict) -> str:
    """The card's tables (markdown): stream maxima, c sweep, quality per runtime,
    adapters, whole-pipeline fp16, file checks."""
    probe, plain = study["probe"], study["plain"]
    c = probe["chosen_c"]
    out = [f"# {variant}: fp16-fix tables", "", "## Max |x| per site", ""]
    sites = list(probe["summary_max"])
    out += md_table(
        ["run"] + sites,
        [
            [name] + [f"{src[s]:,.1f}" if s in src else "" for s in sites]
            for name, src in (
                ("official, probe grid (fp32)", probe["summary_max"]),
                ("official, sampling (fp32)", probe["sampled_summary_max"]),
                ("official, plain fp16", plain["fp16_summary_max"]),
                ("official, bf16", plain["bf16_summary_max"]),
                (f"folded c = 1/{round(1 / c)} (fp32)", probe["folded_summary_max"]),
            )
        ],
    )
    n_rows = len(plain["fp16_forward_rows"])
    bad = {
        k: sum(r["nonfinite"] for r in plain[f"{k}_forward_rows"])
        for k in ("fp16", "bf16")
    }
    out += [
        f"Non-finite forwards on the official weights: fp16 {bad['fp16']} / {n_rows},"
        f" bf16 {bad['bf16']} / {n_rows}.",
        "",
        "## c sweep",
        "",
    ]
    rows = []
    for r in study["sweep"]:
        t = r["weights"]["total"]
        eps = max(q["stock_rel"] for q in study["eps"][str(r["c"])])
        rows.append(
            [
                f"1/{round(1 / r['c'])}" + (" (chosen)" if r["c"] == c else ""),
                f"{r['fp16_stream_max']:,.0f}",
                f"{r['headroom']:.1f}",
                f"{r['nonfinite_forwards']} / {r['forwards']}",
                f"{r['mean_lpips']:.4f}",
                f"{r['mean_psnr']:.2f}",
                f"{r['mean_latent_cos']:.4f}",
                f"{(t['flushed'] + t['subnormal']) / t['n']:.2%}",
                f"{r['stream']['subnormal_frac']:.2e}",
                f"{eps:.1e}",
            ]
        )
    n = len(study["sweep"][0]["vs_fp32"])
    bf = plain["bf16_vs_fp32"][:n]
    rows.append(
        ["official bf16", "", "", "", f"{mean([q['lpips'] for q in bf]):.4f}"]
        + [f"{mean([q['psnr'] for q in bf]):.2f}"]
        + [f"{mean([q['latent_cos'] for q in bf]):.4f}", "", "", ""]
    )
    out += md_table(
        [
            "c",
            "fp16 stream max",
            "headroom",
            "non-finite",
            f"LPIPS vs fp32 ({n} img)",
            "PSNR",
            "latent cos",
            "scaled weights subnormal / 0",
            "stream subnormal",
            "stock-eps rel err",
        ],
        rows,
    )
    for rt in ("kohakumoss", "diffusers", "comfyui"):
        q = quality.get(rt)
        if not q:
            continue
        out += [f"## Quality, {rt} (vs fp32; fix vs bf16 last)", ""]
        rows = [score_row(name, v) for name, v in q["vs_fp32"].items()]
        rows.append(score_row("fix_fp16 vs bf16", q["fix_vs_bf16"]))
        rows += [
            score_row(f"{i} {k}", v[k])
            for i, v in q.get("lora", {}).items()
            for k in ("fix_vs_bf16", "effect_vs_bf16")
        ]
        rows += [
            score_row(f"{i} {name} vs fp32 + adapter", s)
            for i, v in q.get("lora", {}).items()
            for name, s in v.get("vs_fp32", {}).items()
        ]
        out += md_table(["config", "n", "LPIPS", "SSIM", "PSNR"], rows)
    te, vae = study["text_encoder"]["cos"], study["vae"]
    wf = study["whole_fp16"]["vs_fp32"]
    out += ["## Whole pipeline in fp16", ""]
    out += md_table(
        ["part", "metric", "bf16", "fp16"],
        [
            [
                "text encoder",
                "cos vs fp32 (min / mean)",
                f"{min(te['bf16_vs_fp32']):.6f} / {mean(te['bf16_vs_fp32']):.6f}",
                f"{min(te['fp16_vs_fp32']):.6f} / {mean(te['fp16_vs_fp32']):.6f}",
            ],
            [
                "VAE decode",
                "PSNR vs fp32 (mean dB)",
                f"{mean([q['psnr_vs_fp32'] for q in vae['bf16']]):.2f}",
                f"{mean([q['psnr_vs_fp32'] for q in vae['fp16']]):.2f}",
            ],
            [
                "TE + fixed DiT + VAE",
                f"LPIPS / PSNR vs fp32 ({len(wf)} img)",
                "",
                f"{mean([q['lpips'] for q in wf]):.4f} / {mean([q['psnr'] for q in wf]):.2f}",
            ],
        ],
    )
    files = quality.get("files")
    if files:
        out += ["## Files", "", "```", json.dumps(files, indent=1), "```", ""]
    return "\n".join(out)


def main() -> None:
    check_script(globals())
    if not RUNS or not OUT:
        raise SystemExit("set RUNS and OUT")
    runs, out = Path(RUNS), Path(OUT)
    style()
    for variant in VARIANTS:
        study = json.loads((runs / "study" / variant / "report.json").read_text())
        qpath = runs / "quality" / variant / "quality.json"
        quality = json.loads(qpath.read_text()) if qpath.is_file() else {}
        epath = runs / "export" / f"{variant}_export.json"
        export = json.loads(epath.read_text()) if epath.is_file() else {}
        assets = {
            "stream": stream_chart(variant, study, out),
            "plain": plain_chart(variant, study, out),
            "sweep": sweep_chart(variant, study, out),
            "plain_examples": failure(variant, runs, out),
            "sweep_strip": sweep_strip(variant, study, runs, out),
            "gallery": gallery(variant, quality, runs, out) if quality else [],
        }
        merged = {
            "study": study,
            "quality": quality,
            "export": export,
            "assets": assets,
        }
        rep = out / "reports" / f"{variant}_fp16_fix.json"
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(json.dumps(merged, indent=1), encoding="utf-8")
        tab = out / "reports" / f"{variant}_tables.md"
        tab.write_text(tables(variant, study, quality), encoding="utf-8")
        print(f"{variant}: {sum(len(v) for v in assets.values())} assets, {rep}")


if __name__ == "__main__":
    main()
