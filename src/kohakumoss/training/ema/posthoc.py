"""Post-hoc EMA (EDM2): combine power-EMA snapshots into an EMA of any sigma_rel.

A power EMA of exponent g taken at step t has averaging profile
p(tau) = (g+1) tau^g / t^(g+1) on [0, t]. The weights are the least-squares fit of
p_target by the snapshot profiles in L2 over tau, solved in float32 in the basis of
unit-norm profiles, whose Gram matrix has a closed form with O(1) terms. See
docs/concepts/training.md (EMA).

    python -m kohakumoss.training.ema.posthoc --run <output_dir> --sigma-rel 0.1 --out ema.safetensors
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from kohakumoss.training.ema.averagers import gamma_from_sigma_rel


def correlation(t_a: Tensor, g_a: Tensor, t_b: Tensor, g_b: Tensor) -> Tensor:
    """<p_a, p_b> / (|p_a| |p_b|) for power profiles, broadcast, float32:
    sqrt((2 g_a + 1)(2 g_b + 1)) / (g_a + g_b + 1) * (lo/t_a)^(g_a + 1/2)
    * (lo/t_b)^(g_b + 1/2), lo = min(t_a, t_b)."""
    lo = torch.minimum(t_a, t_b)
    log_c = (
        0.5 * torch.log((2 * g_a + 1) * (2 * g_b + 1))
        - torch.log(g_a + g_b + 1)
        + (g_a + 0.5) * torch.log(lo / t_a)
        + (g_b + 0.5) * torch.log(lo / t_b)
    )
    return log_c.exp()


def profile_norm(t: Tensor, g: Tensor) -> Tensor:
    """|p| = (g + 1) / sqrt((2 g + 1) t) in L2 over [0, t]."""
    return (g + 1) / torch.sqrt((2 * g + 1) * t)


def solve_weights(
    snapshots: list[tuple[float, float]], target_t: float, target_gamma: float
) -> Tensor:
    """Least-squares weights [n] (float32) for snapshots [(t, gamma)] approximating the
    profile (target_t, target_gamma): C y = c in the unit-norm basis (SVD solve), then
    x_i = y_i |p_target| / |p_i|."""
    f32 = torch.float32
    t = torch.tensor([s for s, _ in snapshots], dtype=f32)
    g = torch.tensor([gm for _, gm in snapshots], dtype=f32)
    tt = torch.tensor(target_t, dtype=f32)
    gt = torch.tensor(target_gamma, dtype=f32)
    gram = correlation(t[:, None], g[:, None], t[None, :], g[None, :])
    rhs = correlation(t, g, tt, gt)
    y = torch.linalg.lstsq(gram, rhs.unsqueeze(-1), driver="gelsd").solution
    return y.squeeze(-1) * profile_norm(tt, gt) / profile_norm(t, g)


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
