"""Post-hoc EMA (EDM2): combine power-EMA snapshots into an EMA of any sigma_rel.

A power EMA of exponent g taken at step t has averaging profile
p(tau) = (g+1) tau^g / t^(g+1) on [0, t]. Weights x solve A x = b with
A_ij = <p_i, p_j> and b_i = <p_i, p_target> in L2 over tau.

    python -m kohakumoss.training.ema.posthoc --run <output_dir> --sigma-rel 0.1 --out ema.safetensors
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from kohakumoss.training.ema.averagers import gamma_from_sigma_rel


def _inner(t_i: float, g_i: float, t_j: float, g_j: float) -> float:
    lo = min(t_i, t_j)
    return (
        (g_i + 1)
        * (g_j + 1)
        * lo ** (g_i + g_j + 1)
        / ((g_i + g_j + 1) * t_i ** (g_i + 1) * t_j ** (g_j + 1))
    )


def solve_weights(
    snapshots: list[tuple[float, float]], target_t: float, target_gamma: float
) -> Tensor:
    """Least-squares weights for snapshots [(t, gamma)] approximating (target_t, target_gamma)."""
    n = len(snapshots)
    a = torch.empty(n, n, dtype=torch.float64)
    b = torch.empty(n, dtype=torch.float64)
    for i, (ti, gi) in enumerate(snapshots):
        b[i] = _inner(ti, gi, target_t, target_gamma)
        for j, (tj, gj) in enumerate(snapshots):
            a[i, j] = _inner(ti, gi, tj, gj)
    return torch.linalg.lstsq(a, b.unsqueeze(-1)).solution.squeeze(-1)


def combine(states: list[dict[str, Tensor]], weights: Tensor) -> dict[str, Tensor]:
    """sum_i w_i * state_i, per key, in fp32."""
    out = {}
    for key in states[0]:
        acc = torch.zeros_like(states[0][key], dtype=torch.float32)
        for w, st in zip(weights.tolist(), states):
            acc += w * st[key].float()
        out[key] = acc
    return out


def reconstruct(run_dir: str, sigma_rel: float, target_step: int | None = None):
    """Post-hoc EMA from `<run>/ema_snapshots/*.safetensors` (+ `index.json`)."""
    root = Path(run_dir) / "ema_snapshots"
    index = json.loads((root / "index.json").read_text())
    snaps, states = [], []
    for entry in index:
        sd = load_file(str(root / entry["file"]))
        for tag, gamma in entry["gammas"].items():
            snaps.append((float(entry["step"]), float(gamma)))
            states.append(
                {
                    k.split("/", 1)[1]: v
                    for k, v in sd.items()
                    if k.startswith(tag + "/")
                }
            )
    t = float(target_step or max(s for s, _ in snaps))
    weights = solve_weights(snaps, t, gamma_from_sigma_rel(sigma_rel))
    return combine(states, weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--sigma-rel", type=float, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    state = reconstruct(args.run, args.sigma_rel, args.step)
    save_file({k: v.contiguous() for k, v in state.items()}, args.out)


if __name__ == "__main__":
    main()
