"""`kohakumoss-config`: write configs from the presets, list them, check configs.

    kohakumoss-config new --from lora_24g_bf16 my_run.py
    kohakumoss-config wizard
    kohakumoss-config new train --weights <ComfyUI models dir> --data <dataset dir>
    kohakumoss-config new sample --weights <ComfyUI models dir> --adapter <file>.safetensors
    kohakumoss-config list          (the named memory profiles)
    kohakumoss-config list --all    (every catalog profile)
    kohakumoss-config check my_run.py

Hardware flags left out are read from the local GPU. See docs/guides/configs.md.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from kohakumoss.cli.config import hardware, presets
from kohakumoss.cli.config.full import render_full
from kohakumoss.cli.config.starter import (
    DATA_SCRIPTS,
    SampleChoice,
    Starter,
    TrainChoice,
    data_starter,
    posix,
    render,
    sample_starter,
    shown,
    train_starter,
)
from kohakumoss.cli.config.validate import (
    MISSING,
    TRAIN,
    USER_DIR,
    blocking,
    compose,
    config_problems,
    is_checkout,
    problems,
    repo_root,
    script_ref,
    target_script,
)
from kohakumoss.logging.log import get_logger, setup
from kohakumoss.memory.catalog_table import markdown as profile_table
from kohakumoss.memory.profiles import CATALOG
from kohakumoss.memory.profiles import ORDER as PROFILE_ORDER

DEFAULT_PROMPT = "a lighthouse on a rocky coast at sunset"
log = get_logger(__name__)


def _vram(text: str) -> float | str:
    return text if text == presets.AUTO else float(text)


def _tolerance(text: str) -> float:
    value = float(text)
    if not 0.0 <= value < 1.0:
        raise argparse.ArgumentTypeError(f"{text}: a fraction in [0, 1)")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kohakumoss-config", description="KohakuMoss config helpers"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    new = sub.add_parser("new", help="write a config").add_subparsers(
        dest="kind", required=True
    )
    train = new.add_parser("train", help="train config + data-preparation config")
    train.add_argument(
        "--weights",
        required=True,
        help="ComfyUI models folder (diffusion_models/ ...) or HF model dir",
    )
    train.add_argument("--data", required=True, help="dataset root")
    train.add_argument("--source", default="kohya", help="data preset (see `list`)")
    train.add_argument("--algo", default="lora", help="algo preset (see `list`)")
    train.add_argument("--gpus", type=int, default=1, help="devices (Linux for > 1)")
    train.add_argument("--name", default="my-lora", help="RUN_NAME and file name")
    train.add_argument("--steps", type=int, default=0, help="0 = the base STEPS")
    sample = new.add_parser("sample", help="sample config")
    sample.add_argument(
        "--weights",
        required=True,
        help="ComfyUI models folder, HF model dir or weights root",
    )
    sample.add_argument("--adapter", action="append", default=[], help="path[:mult]")
    sample.add_argument("--name", default="sample", help="file name")
    sample.add_argument("--samples", default="runs/samples", help="output directory")
    for p in (train, sample):
        p.add_argument("--prompt", action="append", default=[])
        p.add_argument("--vram", type=_vram, default=None, help="GiB or 'auto'")
        p.add_argument(
            "--vram-tolerance",
            type=_tolerance,
            default=presets.VRAM_TOLERANCE,
            help="fraction of a preset size a card may fall short of and still select it "
            f"(default {presets.VRAM_TOLERANCE}; a 24 GB card reports 23.99 GiB)",
        )
        p.add_argument("--precision", choices=("bf16", "fp16"), default=None)
        p.add_argument("--out", default="", help=f"default {USER_DIR}/<name>.py")
        p.add_argument("--force", action="store_true", help="overwrite")
    sub.add_parser("wizard", help="ask for GPU, algorithm, data and weights")
    listing = sub.add_parser("list", help="list the presets, recipes and profiles")
    listing.add_argument(
        "--all", action="store_true", help="every catalog memory profile"
    )
    check = sub.add_parser("check", help="validate configs through the knob checks")
    check.add_argument("configs", nargs="+")
    return parser


def build_from_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kohakumoss-config new --from", description="a full config from a base"
    )
    parser.add_argument("--from", dest="base", required=True, help="recipe or path")
    parser.add_argument("out", help="config file to write")
    parser.add_argument("--force", action="store_true", help="overwrite")
    return parser


def _hardware(vram, precision) -> tuple[int | str, str, str]:
    """(VRAM GiB, precision, device kind): the given values, the local GPU for the
    rest; a host without a GPU gets device "cuda"."""
    acc = hardware.detect()
    if acc.kind == "cpu":
        if vram is None or precision is None:
            raise SystemExit("no GPU found: pass --vram and --precision")
        return vram, precision, "cuda"
    log.info(
        f"detected {acc.kind}: {acc.vram_gib:.2f} GiB, bf16={acc.bf16}, x{acc.count}"
    )
    vram = vram if vram is not None else acc.vram_gib
    precision = precision or ("bf16" if acc.bf16 else "fp16")
    return vram, precision, acc.kind


def out_dir(root: Path) -> Path:
    """Default directory for written configs: configs/user/ of a checkout, else cwd."""
    return root / USER_DIR if is_checkout(root) else Path.cwd()


def _target(path: Path, force: bool) -> Path:
    if path.exists() and not force:
        raise SystemExit(f"{path} exists (pass --force to overwrite)")
    return path


def _ref(root: Path, path: Path) -> str:
    return shown(path, root) if is_checkout(root) else posix(path)


def _check(root: Path, st: Starter) -> dict:
    """Composed knobs of `st`; SystemExit on blocking problems, missing paths warn."""
    knobs = compose(root, st.script, st.layers, st.own)
    found = problems(st.script, knobs)
    bad = blocking(found)
    if bad:
        raise SystemExit("the composed config is invalid:\n  - " + "\n  - ".join(bad))
    for p in found:
        log.warning(p)
    return knobs


def _save(root: Path, out: Path, text: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    log.info(f"wrote {_ref(root, out)}")


def write_train(root: Path, choice: TrainChoice, out: Path, force: bool) -> None:
    """Full train config + data config for `choice`; prints the commands to run."""
    out, data_out = _target(out, force), _target(
        out.with_name(f"{out.stem}_data.py"), force
    )
    data_cmds = [
        f"kogine run {script_ref(root, s)} --config {_ref(root, data_out)}"
        for s in DATA_SCRIPTS
    ]
    train_cmd = f"kogine run {script_ref(root, TRAIN)} --config {_ref(root, out)}"
    commands = data_cmds + [train_cmd]
    st = train_starter(root, choice)
    _save(root, out, render_full(st, out, commands, root, _check(root, st)))
    data = data_starter(choice, out)
    _check(root, data)
    _save(root, data_out, render(data, out=data_out, commands=commands, root=root))
    log.info("next:\n  " + "\n  ".join(commands))


def new_train(root: Path, args) -> None:
    vram, precision, device = _hardware(args.vram, args.precision)
    choice = TrainChoice(
        weights=args.weights,
        data=args.data,
        source=args.source,
        algo=args.algo,
        vram=vram,
        precision=precision,
        tolerance=args.vram_tolerance,
        gpus=args.gpus,
        device=device,
        name=args.name,
        steps=args.steps,
        prompts=tuple(args.prompt),
    )
    out = Path(args.out) if args.out else out_dir(root) / f"{args.name}.py"
    write_train(root, choice, out, args.force)


def base_path(root: Path, name: str) -> Path:
    """A recipe / quality / base name, or a config path."""
    moss = root / "configs" / "moss"
    for path in (
        Path(name),
        moss / "recipes" / f"recipes_{name}.py",
        moss / "recipes" / f"{name}.py",
        moss / "quality" / f"{name}.py",
        moss / "_base" / f"{name}.py",
        moss / "_base" / f"base_{name}.py",
    ):
        if path.is_file():
            return path.resolve()
    have = ", ".join(recipes(root))
    raise SystemExit(f"no base config {name!r}; recipes: {have}")


def recipes(root: Path) -> list[str]:
    folder = root / "configs" / "moss" / "recipes"
    return sorted(p.stem.removeprefix("recipes_") for p in folder.glob("recipes_*.py"))


def new_from(root: Path, base: str, out: Path, force: bool) -> None:
    """A full train config that extends base config `base`."""
    path = base_path(root, base)
    if target_script(path) != TRAIN:
        raise SystemExit(f"{path} does not drive {TRAIN}")
    out = _target(out, force)
    title = f"{out.stem}: extends {path.stem}. Drives {TRAIN}."
    st = Starter(TRAIN, title, [path], {})
    command = f"kogine run {script_ref(root, TRAIN)} --config {_ref(root, out)}"
    _save(root, out, render_full(st, out, [command], root, _check(root, st)))
    log.info(f"edit WEIGHTS and DATASETS, then:\n  {command}")


def new_sample(root: Path, args) -> None:
    vram, precision, device = _hardware(args.vram, args.precision)
    if vram == presets.AUTO:
        raise SystemExit("sampling needs --vram in GiB, not 'auto'")
    if not args.prompt:
        log.warning(
            f"no --prompt: the sample config's PROMPT is the placeholder "
            f"{DEFAULT_PROMPT!r}; set it to a caption of your data"
        )
    choice = SampleChoice(
        weights=args.weights,
        adapters=tuple(args.adapter),
        prompts=tuple(args.prompt) or (DEFAULT_PROMPT,),
        vram=vram,
        precision=precision,
        tolerance=args.vram_tolerance,
        device=device,
        out=args.samples,
    )
    out = _target(
        Path(args.out) if args.out else out_dir(root) / f"{args.name}.py", args.force
    )
    st = sample_starter(root, choice)
    _check(root, st)
    command = f"kogine run {script_ref(root, st.script)} --config {_ref(root, out)}"
    _save(root, out, render(st, out, [command], root))
    log.info(f"next:\n  {command}")


def _ask(ask: Callable[[str], str], question: str, default: str = "") -> str:
    """One answer; empty input takes `default`; asks again while both are empty."""
    while True:
        shown_default = f" [{default}]" if default else ""
        answer = ask(f"{question}{shown_default}: ").strip() or default
        if answer:
            return answer


def _choose(ask, question: str, options: list[str], default: str) -> str:
    while True:
        answer = _ask(ask, f"{question} ({' / '.join(options)})", default)
        if answer in options:
            return answer
        print(f"choose one of: {', '.join(options)}")


def wizard(root: Path, ask: Callable[[str], str] = input) -> None:
    """Ask for GPU, precision, algorithm, data and weights; write the train configs."""
    acc = hardware.detect()
    found = f"{acc.vram_gib:.2f}" if acc.vram_gib else "24"
    vram = _ask(ask, "GPU memory in GiB, or auto", found)
    bf16 = "bf16" if acc.bf16 or acc.kind == "cpu" else "fp16"
    precision = _choose(ask, "precision", ["bf16", "fp16"], bf16)
    algo = _choose(ask, "algorithm", presets.names(root, "moss", "algo"), "lora")
    source = _choose(
        ask, "dataset layout", presets.names(root, "moss", "data"), "kohya"
    )
    data = _ask(ask, "dataset path")
    weights = _ask(
        ask,
        "weights path (a ComfyUI models folder with diffusion_models/, "
        "text_encoders/, vae/, or an HF dir with transformer/, text_encoder/, vae/)",
    )
    steps = int(_ask(ask, "steps (0 = the base value)", "0"))
    gpus = int(_ask(ask, "GPUs", "1"))
    name = _ask(ask, "run name", "my-lora")
    out = Path(_ask(ask, "config file", posix(out_dir(root) / f"{name}.py")))
    choice = TrainChoice(
        weights=weights,
        data=data,
        source=source,
        algo=algo,
        vram=_vram(vram),
        precision=precision,
        gpus=gpus,
        device=acc.kind if acc.kind != "cpu" else "cuda",
        name=name,
        steps=steps,
    )
    write_train(root, choice, out, force=out.exists() and _confirm(ask, out))


def _confirm(ask, out: Path) -> bool:
    if _ask(ask, f"{out} exists; overwrite? (y/n)", "n").lower() != "y":
        raise SystemExit("nothing written")
    return True


def list_presets(root: Path, everything: bool = False) -> None:
    """Recipes, config layers and the memory profiles: the named ones, or with
    `everything` every catalog entry."""
    print(f"recipes          {', '.join(recipes(root))}")
    for domain, groups in presets.catalogue(root).items():
        for group, items in groups.items():
            print(f"{domain}/{group:10s} {', '.join(items)}")
    print("\nmemory profiles (MEMORY_PROFILE; estimated peak at 1 MP, LoRA rank 16):")
    print(profile_table() if everything else profile_table(names=PROFILE_ORDER))
    if not everything:
        print(
            f"\n{len(CATALOG)} catalog profiles more: kohakumoss-config list --all "
            "(docs/concepts/profiles.md)"
        )


def check_configs(root: Path, paths: list[str]) -> int:
    """Print every problem per config; returns the number of configs with problems."""
    bad = 0
    for path in paths:
        found = config_problems(Path(path), root)
        print(f"{path}: {'ok' if not found else 'invalid'}")
        for p in found:
            print(f"  - {p}" + ("  (path)" if MISSING in p else ""))
        bad += bool(found)
    return bad


def main(argv: list[str] | None = None) -> None:
    setup()
    argv = sys.argv[1:] if argv is None else list(argv)
    root = repo_root()
    if argv[:1] == ["new"] and "--from" in argv:
        args = build_from_parser().parse_args(argv[1:])
        new_from(root, args.base, Path(args.out), args.force)
        return
    args = build_parser().parse_args(argv)
    match args.command, getattr(args, "kind", None):
        case "new", "train":
            new_train(root, args)
        case "new", "sample":
            new_sample(root, args)
        case "wizard", _:
            wizard(root)
        case "list", _:
            list_presets(root, args.all)
        case "check", _:
            sys.exit(1 if check_configs(root, args.configs) else 0)


if __name__ == "__main__":
    main()
