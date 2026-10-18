"""The local side of the bench: freeze the working tree, ship it to each host,
install it into the host's venv, start the host runner, read its status and collect its
results. Transports: "ssh" (a command prefix from the hosts file), "local" (this
machine, the host runner in the foreground) and "manual" (print the commands).
See docs/performance/bench-matrix.md#launch.
"""

import hashlib
import io
import json
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

from kohakumoss_bench.plan.config import Host

SKIP = ("runs/", "out/", ".venv")  # outputs; local-only files stay out via .gitignore
MARK = ".bench-tree"
HOST_SCRIPT = "scripts/bench/matrix/host.py"


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, check=True
    ).stdout


def tree_files(repo: Path) -> list[str]:
    """Tracked and untracked (not ignored) files that exist, minus local-only paths."""
    listed = _git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    names = sorted({n for n in listed.decode().split("\0") if n})
    return [
        n
        for n in names
        if (repo / n).is_file() and not any(n.startswith(s) for s in SKIP)
    ]


def tree_identity(repo: Path, files: list[str]) -> str:
    """`<HEAD short>-<sha1 of every shipped file's path and bytes>`."""
    head = _git(repo, "rev-parse", "--short=10", "HEAD").decode().strip()
    digest = hashlib.sha1()
    for name in files:
        digest.update(name.encode() + b"\0")
        digest.update((repo / name).read_bytes())
    return f"{head}-{digest.hexdigest()[:10]}"


def tree_tar(repo: Path, files: list[str], identity: str) -> bytes:
    """The shipped files plus the identity marker as one tar."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name in files:
            tar.add(repo / name, arcname=name, recursive=False)
        info = tarfile.TarInfo(MARK)
        data = identity.encode()
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def tree_dir(host: Host, identity: str, name: str = "") -> str:
    """The host's directory of a frozen tree: `trees/<identity>` for the main repo,
    `trees/<name>-<identity>` for an extra repo."""
    return f"{host.root}/trees/{name + '-' if name else ''}{identity}"


def _ssh(host: Host, command: str, data: bytes | None = None, check: bool = True):
    return subprocess.run(
        [*host.ssh, command], input=data, capture_output=True, check=check
    )


def push(
    host: Host, repo: Path, files: list[str], identity: str, name: str = ""
) -> str:
    """Ship a frozen tree to the host (skipped when that identity is there)."""
    target = tree_dir(host, identity, name)
    q = shlex.quote(target)
    match host.transport:
        case "ssh":
            if _ssh(host, f"test -f {q}/{MARK}", check=False).returncode == 0:
                return target
            tar = tree_tar(repo, files, identity)
            _ssh(host, f"mkdir -p {q} && tar -xf - -C {q}", data=tar)
        case "local":
            if (Path(target) / MARK).is_file():
                return target
            with tarfile.open(fileobj=io.BytesIO(tree_tar(repo, files, identity))) as t:
                t.extractall(target)
        case "manual":
            out = Path("runs") / "bench" / f"tree-{identity}.tar"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(tree_tar(repo, files, identity))
            print(f"[{host.name}] copy {out} to the host, untar into {target}")
    return target


def install(host: Host, tree: str) -> None:
    """The host's install command over the frozen tree (editable, no dependencies);
    an empty `install` skips it."""
    if not host.install:
        return
    command = host.install.format(python=host.python, tree=tree)
    match host.transport:
        case "ssh":
            _ssh(host, command)
        case "local":
            subprocess.run(shlex.split(command, posix=False), check=True)
        case "manual":
            print(f"[{host.name}] {command}")


def config_path(host: Host, bench: str) -> str:
    return f"{host.root}/configs/{bench}-{host.name}.json"


def put_config(host: Host, bench: str, config: dict) -> str:
    """Write the host runner's JSON `config` on `host`; its path there."""
    path = config_path(host, bench)
    data = json.dumps(config, indent=1).encode()
    match host.transport:
        case "ssh":
            q = shlex.quote(path)
            _ssh(host, f"mkdir -p $(dirname {q}) && cat > {q}", data=data)
        case "local":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(data)
        case "manual":
            out = Path("runs") / "bench" / f"{bench}-{host.name}.json"
            out.write_bytes(data)
            print(f"[{host.name}] copy {out} to {path}")
    return path


def start(host: Host, tree: str, config: str, bench: str) -> None:
    """Start the host runner: detached over ssh, in the foreground locally."""
    log = f"{host.root}/runs/{bench}/{host.name}.log"
    script = f"{tree}/{HOST_SCRIPT}"
    match host.transport:
        case "ssh":
            q = shlex.quote
            command = (
                f"mkdir -p $(dirname {q(log)}) && cd {q(tree)} && (nohup "
                f"{q(host.python)} {q(script)} --config {q(config)} > {q(log)} 2>&1 "
                "< /dev/null &)"
            )
            _ssh(host, command)
        case "local":
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            with open(log, "w", encoding="utf-8") as f:
                subprocess.run(
                    [host.python, script, "--config", config],
                    cwd=tree,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        case "manual":
            print(
                f"[{host.name}] cd {tree} && {host.python} {script} --config {config}"
            )


def join(host: Host, bench: str, gpus: list[int]) -> str:
    """Write `join.json` ({"gpus": gpus}) into the running bench's result directory:
    the host runner starts those cards in their pools once they are idle."""
    path = f"{host.root}/runs/{bench}/{host.name}/join.json"
    data = json.dumps({"gpus": gpus}).encode()
    match host.transport:
        case "ssh":
            _ssh(host, f"cat > {shlex.quote(path)}", data=data)
        case "local":
            Path(path).write_bytes(data)
        case "manual":
            print(f"[{host.name}] write {data.decode()} to {path}")
    return path


def preflight(host: Host, checks: dict[str, str]) -> dict[str, str]:
    """Read-only shell checks on an ssh host: name -> "ok" / "FAIL" plus the first
    output line; other transports report nothing."""
    out = {}
    if host.transport != "ssh":
        return out
    for name, command in checks.items():
        done = _ssh(host, command, check=False)
        text = (done.stdout or done.stderr).decode(errors="replace").strip()
        line = text.splitlines()[0] if text else ""
        out[name] = ("ok" if done.returncode == 0 else "FAIL") + (
            f": {line[:160]}" if line else ""
        )
    return out


def status(host: Host, bench: str) -> dict:
    path = f"{host.root}/runs/{bench}/{host.name}/status.json"
    match host.transport:
        case "ssh":
            done = _ssh(host, f"cat {shlex.quote(path)}", check=False)
            return json.loads(done.stdout) if done.returncode == 0 else {}
        case "local":
            p = Path(path)
            return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    return {}


def collect(host: Host, bench: str, into: Path) -> Path:
    """Copy the host's result directory to `into/<host>/`."""
    src = f"{host.root}/runs/{bench}/{host.name}"
    dst = into / host.name
    dst.mkdir(parents=True, exist_ok=True)
    match host.transport:
        case "ssh":
            data = _ssh(host, f"tar -cf - -C {shlex.quote(src)} .").stdout
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                tar.extractall(dst)
        case "local":
            shutil.copytree(src, dst, dirs_exist_ok=True)
        case "manual":
            print(f"[{host.name}] copy {src} to {dst}")
    return dst
