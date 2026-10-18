"""The prompt list of a comparison grid, written to OUT as JSON (a valid Python literal):
per artist tag of ARTISTS, the PROMPTS bodies, or CAPTIONS captions read from that
artist's folder of a kohya-style DATASET (`<repeats>_..._<artist>/*.txt`, picked with
SEED). scripts/examples/grid.ps1 and grid.sh write its config and read OUT.

    kogine run scripts/examples/grid_prompts.py --config <prompts config>
    kogine run scripts/examples/grid_prompts.py --set PROMPTS=["a cat"] --set OUT=prompts.json
"""

import json
import random
import re
from pathlib import Path
from typing import Annotated

from kohakuengine import FormatArg

from kohakumoss_common.knobs import check_script, prompt_list

DATASET = ""  # kohya-style dataset root (read when PROMPTS is empty)
ARTISTS = []  # artist tags, as folder names without the numeric prefix; "" = no tag
PROMPTS: Annotated[list, FormatArg(prompt_list)] = (
    []
)  # prompt bodies; empty = CAPTIONS captions per artist from DATASET; a bare string is one body
CAPTIONS = 1  # captions per artist
SEED = 0  # picks the captions
SUFFIX = ""  # text appended to every prompt
OUT = ""  # JSON file the prompt list is written to


def captions(dataset: Path, artist: str, count: int, seed: int) -> list[str]:
    """`count` seeded tag lists from the folder `<repeats>_..._<artist>` of `dataset`."""
    pattern = re.compile(r"(\d+_)+" + re.escape(artist))
    folders = [
        p for p in sorted(dataset.iterdir()) if p.is_dir() and pattern.fullmatch(p.name)
    ]
    if not folders:
        raise SystemExit(f"no folder <repeats>_..._{artist} in {dataset}")
    texts = sorted(folders[0].glob("*.txt"))
    if not texts:
        raise SystemExit(f"no .txt captions in {folders[0]}")
    picks = random.Random(seed).sample(texts, min(count, len(texts)))
    return [p.read_text(encoding="utf-8").strip() for p in picks]


def prompts(k: dict) -> list[str]:
    """`<artist>, <body>[, <suffix>]` per artist and body (artist underscores as spaces)."""
    out = []
    for artist in k["ARTISTS"] or [""]:
        bodies = list(k["PROMPTS"])
        if not bodies:
            if not artist:
                raise SystemExit("set PROMPTS, or ARTISTS with DATASET")
            if not k["DATASET"]:
                raise SystemExit(f"artist {artist!r} needs DATASET to read captions")
            bodies = captions(Path(k["DATASET"]), artist, k["CAPTIONS"], k["SEED"])
        tag = artist.replace("_", " ")
        for body in bodies:
            text = f"{tag}, {body}" if tag else body
            out.append(f"{text}, {k['SUFFIX']}" if k["SUFFIX"] else text)
    return out


def main() -> None:
    k = check_script(globals())
    if not k["OUT"]:
        raise SystemExit("set OUT, the JSON file to write")
    Path(k["OUT"]).write_text(json.dumps(prompts(k)), encoding="utf-8")


if __name__ == "__main__":
    main()
