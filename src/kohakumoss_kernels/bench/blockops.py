"""Bench harness of the block ops: backend vs eager composition vs torch.compile of it,
forward and forward + backward time and error statistics against fp64, rows to
`runs/kernel/<op>-<card>.json`. See docs/internals/kernel-dev.md#benches.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from kohakumoss_kernels import compile_setup, registry
from kohakumoss_kernels.backends.compile.ops import COMPILED, compiled_op
from kohakumoss_kernels.error_stats import stats, worse
from kohakumoss_kernels.hardware.device import measure_rates
from kohakumoss_kernels.tuning.timing import best_ms, graph_ms, l2_flush

__all__ = ["contenders", "dtypes", "measure", "parse", "report", "segment_ids"]


def segment_ids(lengths: list[int], device) -> torch.Tensor:
    """Per-token sample index (int32 [sum(lengths)]) of samples packed in order."""
    return torch.cat(
        [torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lengths)]
    ).to(device)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="auto", help="triton | tilelang | auto")
    p.add_argument("--dtypes", default="float16,bfloat16")
    p.add_argument("--out", default="runs/kernel")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--device", default="cuda", help="cuda | cuda:N | xpu:N")
    return p.parse_args()


def dtypes(args) -> list[torch.dtype]:
    return [getattr(torch, d) for d in args.dtypes.split(",")]


def _leaves(args: Sequence, grads: Sequence[bool], to64: bool = False) -> list:
    out = []
    for a, g in zip(args, grads):
        if isinstance(a, torch.Tensor):
            t = a.detach()
            t = t.to(torch.float64) if to64 and t.is_floating_point() else t.clone()
            out.append(t.requires_grad_(True) if g else t)
        else:
            out.append(a)
    return out


def _tuple(x) -> tuple:
    return x if isinstance(x, tuple) else (x,)


def _grads(leaves: list, grads: Sequence[bool]) -> list:
    return [leaves[i].grad for i, g in enumerate(grads) if g]


def _graph_or_nan(fn: Callable, device: torch.device) -> float:
    """Graph-replayed device ms of a forward + backward step (nan if capture fails)."""
    try:
        return graph_ms(fn, device)
    except Exception:  # noqa: BLE001
        return float("nan")


def measure(
    name: str,
    fns: dict[str, Callable],
    reference: Callable,
    args: Sequence,
    grads: Sequence[bool],
    dtype: torch.dtype,
    iters: int,
    device: torch.device,
) -> dict:
    """Timing + accuracy of every entry of `fns` on one case."""
    flush = l2_flush(device, measure_rates(device).l2_bytes)
    with torch.no_grad():
        shapes = _tuple(reference(*args))
    gouts = [torch.randn(o.shape, device=device).to(o.dtype) for o in shapes]
    ref_leaves = _leaves(args, grads, True)
    ref_outs = _tuple(reference(*ref_leaves))
    torch.autograd.backward(ref_outs, [g.double() for g in gouts])
    refs = [o.detach() for o in ref_outs] + _grads(ref_leaves, grads)
    row: dict = {"case": name, "dtype": str(dtype).split(".")[-1]}
    selector = fns.get("_selector")
    if selector is not None:
        try:
            row["selected"] = selector.decide(*_leaves(args, grads))
        except Exception as err:  # noqa: BLE001
            row["selected"] = f"error: {type(err).__name__}: {err}"[:200]
    results: dict[str, list[dict]] = {}
    for label, fn in fns.items():
        if label == "_selector":
            continue
        try:
            leaves = _leaves(args, grads)
            outs = _tuple(fn(*leaves))
            torch.autograd.backward(outs, [g.clone() for g in gouts])
            got = [o.detach() for o in outs] + _grads(leaves, grads)
            results[label] = [stats(t, r) for t, r in zip(got, refs)]
            del got, outs, leaves
            row[f"{label}.stats"] = results[label]
            with torch.no_grad():
                row[f"{label}.fwd_ms"] = best_ms(
                    lambda: fn(*args), device, iters, flush
                )
                row[f"{label}.fwd_graph_ms"] = graph_ms(lambda: fn(*args), device)
            fb = _leaves(args, grads)
            local = [g.clone() for g in gouts]

            def step():
                for leaf in fb:
                    if isinstance(leaf, torch.Tensor):
                        leaf.grad = None
                torch.autograd.backward(_tuple(fn(*fb)), local)

            row[f"{label}.fwdbwd_ms"] = best_ms(step, device, iters, flush)
            row[f"{label}.fwdbwd_graph_ms"] = _graph_or_nan(step, device)
        except Exception as err:  # noqa: BLE001
            row[f"{label}.error"] = f"{type(err).__name__}: {err}"[:300]
    if "eager" in results:
        for label, got in results.items():
            tables = [{"kernel": k, "eager": e} for k, e in zip(got, results["eager"])]
            row[f"{label}.worse"] = [worse(t) for t in tables]
    for base in ("eager", "compile", "compile_dyn"):
        if f"{base}.fwd_ms" in row and "backend.fwd_ms" in row:
            for m in ("fwd", "fwdbwd", "fwd_graph", "fwdbwd_graph"):
                row[f"backend.{m}_x_{base}"] = (
                    row[f"{base}.{m}_ms"] / row[f"backend.{m}_ms"]
                )
    return row


def contenders(
    op: str, backend: str, eager_fn: Callable, device, dtype, use_compile: bool = True
) -> dict[str, Callable]:
    """The backend's kernel, the eager composition, torch.compile of it (static
    shapes) and the registered compile path (dynamic shapes); a registered entry that
    chooses per shape is kept under `_selector` for the selection column."""
    registered = registry.resolve(
        op, device, dtype, backend=None if backend == "auto" else backend
    )
    fns = {
        "backend": getattr(registered, "__wrapped__", registered),
        "eager": eager_fn,
    }
    if use_compile:
        torch._dynamo.reset()
        compile_setup.apply()
        fns["compile"] = torch.compile(eager_fn, dynamic=False)
        if op in COMPILED:
            fns["compile_dyn"] = compiled_op(op)
    if hasattr(registered, "decide"):
        fns["_selector"] = registered
    return fns


def report(op: str, rows: list[dict], out_dir: str, device: torch.device) -> Path:
    card = measure_rates(device).key
    path = Path(out_dir) / f"{op}-{card}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"op": op, "card": card, "rows": rows}, indent=1), encoding="utf-8"
    )
    print(f"== {op} on {card}")
    for r in rows:
        cells = [r["case"], r["dtype"], f"selected={r.get('selected', '-')}"]
        for label in ("backend", "eager", "compile", "compile_dyn"):
            if f"{label}.fwd_ms" in r:
                cells.append(
                    f"{label} dev {r[f'{label}.fwd_graph_ms']:.3f}/{r[f'{label}.fwdbwd_graph_ms']:.3f}"
                    f" wall {r[f'{label}.fwd_ms']:.3f}/{r[f'{label}.fwdbwd_ms']:.3f}ms "
                    f"worse {sum(map(bool, r.get(f'{label}.worse', [])))}"
                )
            elif f"{label}.error" in r:
                cells.append(f"{label} ERROR {r[f'{label}.error'][:120]}")
        for base in ("eager", "compile", "compile_dyn"):
            if f"backend.fwd_x_{base}" in r:
                cells.append(
                    f"x{base} dev {r[f'backend.fwd_graph_x_{base}']:.2f}/"
                    f"{r[f'backend.fwdbwd_graph_x_{base}']:.2f} wall "
                    f"{r[f'backend.fwd_x_{base}']:.2f}/{r[f'backend.fwdbwd_x_{base}']:.2f}"
                )
        print(" | ".join(cells))
    return path
