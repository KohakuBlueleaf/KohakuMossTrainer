"""Heal measurements: held-out velocity error and flow loss per sigma, per-block stream
stats of student and (shrunk) teacher, gain and writer drift, and the per-linear
gradient map of student vs teacher."""

import numpy as np
import torch

from kohakumoss.heal.channels import stream_stats
from kohakumoss.heal.losses import rel_sq

SIGMAS = (0.1, 0.3, 0.5, 0.7, 0.9)
BANDS = 6
KINDS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


def band(i: int, depth: int) -> int:
    return min(BANDS - 1, i * BANDS // depth)


class BlockOutputs:
    """Forward hooks keeping each joint block's output while `mode` is "keep" (with
    its graph) or "detach"; "off" keeps nothing (checkpoint recomputes included)."""

    def __init__(self, model):
        self.out: dict[int, torch.Tensor] = {}
        self.mode = "off"
        for i, b in enumerate(model.layers):
            b.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def hook(mod, inputs, output):
            match self.mode:
                case "keep":
                    self.out[i] = output
                case "detach":
                    self.out[i] = output.detach()

        return hook

    def clear(self) -> None:
        self.out = {}

    def ordered(self) -> list[torch.Tensor]:
        return [self.out[i] for i in sorted(self.out)]


class LinearGrads:
    """Per joint-block linear ||dL/dW||_F = ||X^T dY||_F (token Gram matrices) of one
    backward, while `on`."""

    def __init__(self, model):
        self.norm: dict[str, float] = {}
        self.on = False
        for name, mod in model.named_modules():
            parts = name.split(".")
            if (
                len(parts) == 4
                and parts[0] == "layers"
                and parts[2] == "proj"
                and parts[3] in KINDS
            ):
                mod.register_forward_hook(self._forward(name))

    def _forward(self, name):
        def hook(mod, inputs, output):
            if not self.on or not output.requires_grad:
                return
            x = inputs[0].detach()

            def grad(dy):
                x2 = x.reshape(-1, x.shape[-1]).float()
                g2 = dy.detach().reshape(-1, dy.shape[-1]).float()
                self.norm[name] = float(
                    ((x2 @ x2.T) * (g2 @ g2.T)).sum().clamp_min(0).sqrt()
                )

            output.register_hook(grad)

        return hook


@torch.no_grad()
def evaluate(heal, step: int) -> dict:
    """One eval record: per sigma the median / mean velocity error against the teacher
    and both flow losses; per block the student's and the D-scaled teacher's stream
    stats and their relative error; gain drift; writer norms (full_writers)."""
    m = heal.m
    depth = m.student.cfg.depth
    res = {"step": step, "anneal_t": heal.t, "sigma": {}}
    blocks_s, blocks_t, feat = (
        np.zeros((depth, 4)),
        np.zeros((depth, 4)),
        np.zeros(depth),
    )
    count = 0
    for s, est in heal.esteps.items():
        est.noise_gen.manual_seed(1234)
        errs, fs, ft = [], [], []
        for r in heal.held:
            p, v_s, v_t = heal.forward_pair(
                heal.batch([r], [False]), 0, est, keep="detach"
            )
            errs.append(float(rel_sq(v_s, v_t).sqrt()))
            fs.append(float(rel_sq(v_s, p.target)))
            ft.append(float(rel_sq(v_t, p.target)))
            for i in range(depth):
                teacher = heal.tb.out[i].float() * m.d
                blocks_s[i] += stream_stats(heal.sb.out[i], m.massive)
                blocks_t[i] += stream_stats(teacher, m.massive)
                feat[i] += float(rel_sq(heal.sb.out[i], teacher))
            count += 1
        res["sigma"][str(s)] = {
            "v_rel_err_median": float(np.median(errs)),
            "v_rel_err_mean": float(np.mean(errs)),
            "flow_loss_student": float(np.mean(fs)),
            "flow_loss_teacher": float(np.mean(ft)),
        }
    heal.sb.clear()
    heal.tb.clear()
    res["blocks"] = {
        "student": (blocks_s / count).tolist(),
        "teacher_scaled": (blocks_t / count).tolist(),
        "feature_err": (feat / count).tolist(),
        "fields": ["rms", "absmax", "top8_share", "massive_share"],
    }
    res["gain_drift"] = [
        [
            p,
            float(
                torch.cat([t.detach().float().flatten() for t in heal.deltas[p]]).norm()
                / heal.norm_init[p].norm().clamp_min(1e-30)
            ),
        ]
        for p in heal.paths
    ]
    res["writer_norm"] = [
        [n, heal.writer_init[n], float(p.detach().float().norm())]
        for n, p in heal.writers
    ]
    return res


def gradient_map(heal, tag: str, n: int = 8) -> dict:
    """||dL/dW|| per joint-block linear under the flow loss on n held-out samples at
    sigma 0.3 / 0.7, student and teacher; the depth-band x kind medians of their ratio.
    """
    m = heal.m
    out = {"student": [], "teacher": []}
    for s in (0.3, 0.7):
        est = heal.esteps[s]
        for which, model, taps in (
            ("student", m.student, heal.sgrads),
            ("teacher", m.teacher, heal.tgrads),
        ):
            est.noise_gen.manual_seed(99)
            for r in heal.held[:n]:
                p = est.noise(heal.batch([r], [False]), 0)
                z = p.z.detach().clone().requires_grad_(True)
                taps.norm, taps.on = {}, True
                with m.autocast():
                    v = model(z, p.text, p.sigma, p.seq)
                rel_sq(v, p.target).backward()
                taps.on = False
                out[which].append(dict(taps.norm))
                del v, z, p
                model.zero_grad(set_to_none=True)
    depth = m.student.cfg.depth
    ratio = np.full((BANDS, len(KINDS)), np.nan)
    for b in range(BANDS):
        for j, kind in enumerate(KINDS):
            vals = []
            for a, t in zip(out["student"], out["teacher"]):
                for name, g in a.items():
                    parts = name.split(".")
                    if (
                        parts[3] == kind
                        and band(int(parts[1]), depth) == b
                        and t.get(name, 0) > 0
                    ):
                        vals.append(g / t[name])
            if vals:
                ratio[b, j] = float(np.median(vals))
    return {
        "tag": tag,
        "kinds": KINDS,
        "ratio_student_over_teacher": ratio.tolist(),
        "raw": out,
    }
