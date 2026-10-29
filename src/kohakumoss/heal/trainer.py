"""The heal run: the student learns the teacher's velocity on the heal records (plus the
optional hard and concentration terms) while the anneal moves its norms from exact to
plain; data-parallel over the spawned ranks (gradients all-reduced per step). Every
measurement goes to the run directory: metrics.jsonl (per step), evals.jsonl (held-out,
step 0 and every EVAL_EVERY), grads.json (gradient map at the start and the end), parts
(snapshots and the final state)."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from kohakumoss.diffusion.objective import TokenNormalizer
from kohakumoss.heal.config import HealConfig
from kohakumoss.heal.evaluate import (
    SIGMAS,
    BlockOutputs,
    LinearGrads,
    band,
    evaluate,
    gradient_map,
)
from kohakumoss.heal.losses import HealLoss
from kohakumoss.heal.parts import save_parts
from kohakumoss.heal.records import load_records, make_batch
from kohakumoss.heal.sites import model_sites, norm_path
from kohakumoss.heal.student import build_models, stored_gain, train_writers
from kohakumoss.heal.wnorm import anneal_t
from kohakumoss.logging.log import get_logger, setup
from kohakumoss.training.build.loop import build_step
from kohakumoss.training.callbacks.evaluate import FixedSigma
from kohakumoss.training.optim.optimizers.adamw16 import AdamW16

log = get_logger("heal")


def lr_scale(step: int, total: int, warmup: int) -> float:
    """Linear warmup, then cosine to 1 %."""
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))


class Heal:
    def __init__(
        self,
        c: HealConfig,
        k: dict,
        k_t: dict,
        device: torch.device,
        rank: int,
        world: int,
    ):
        self.c, self.device, self.rank, self.world = c, device, rank, world
        self.out = Path(c.out)
        self.m = build_models(c, k, k_t, device)
        self.t = 0.0 if c.anneal_steps > 0 else 1.0
        s = self.m.student
        self.rope, self.depth = s.cfg.rope_axes, s.cfg.depth
        norm = TokenNormalizer()
        self.tstep = build_step(
            k,
            self.m.built,
            norm,
            self.m.autocast,
            device,
            data_rank=rank,
            penalty=False,
        )
        self.esteps = {
            x: build_step(
                k,
                self.m.built,
                norm,
                self.m.autocast,
                device,
                FixedSigma(x),
                penalty=False,
            )
            for x in SIGMAS
        }
        self.loss = HealLoss(c.hard_weight, c.conc_weight, c.conc_k)
        self.sb, self.tb = BlockOutputs(s), BlockOutputs(self.m.teacher)
        self.sgrads, self.tgrads = LinearGrads(s), LinearGrads(self.m.teacher)
        self.paths = [norm_path(x) for x in model_sites(s)]
        self.deltas = {
            p: list(s.get_submodule(p).parametrizations.weight[0].parameters())
            for p in self.paths
        }
        self.norm_init = {
            p: stored_gain(s, p).detach().float().clone() for p in self.paths
        }
        self.writers = train_writers(s) if c.mode == "full_writers" else []
        self.writer_init = {
            n: float(p.detach().float().norm()) for n, p in self.writers
        }
        self.opts = self._optimizers()
        self.trained = [
            p for o in self.opts for g in o.param_groups for p in g["params"]
        ]
        self.base_lr = [[g["lr"] for g in o.param_groups] for o in self.opts]
        self.held, self.train, self.empty = load_records(
            Path(c.records), c.held, rank, world
        )
        n = sum(p.numel() for p in self.trained) / 1e6
        log.info(
            f"rank {rank}: {len(self.train)} train / {len(self.held)} held-out records, {n:.1f}M trained"
        )

    def _optimizers(self) -> list[torch.optim.Optimizer]:
        """Norm deltas and LoKr in AdamW (fp32), full writers in AdamW16."""
        norm_ids = {id(t) for ts in self.deltas.values() for t in ts}
        norms = [t for ts in self.deltas.values() for t in ts]
        lokr = [
            p
            for p in self.m.built.trainable.trainable_params()
            if id(p) not in norm_ids
        ]
        opts = [
            torch.optim.AdamW(
                [{"params": norms, "lr": self.c.lr_norm, "weight_decay": 0.0}],
                betas=(0.9, 0.99),
            )
        ]
        opts.append(
            torch.optim.AdamW(
                [{"params": lokr, "lr": self.c.lr_lokr, "weight_decay": 0.0}],
                betas=(0.9, 0.99),
            )
        )
        if self.writers:
            opts.append(
                AdamW16(
                    [{"params": [p for _, p in self.writers], "lr": self.c.lr_full}],
                    betas=(0.9, 0.99),
                    rounding="stochastic",
                )
            )
        return opts

    def batch(self, recs, drop):
        return make_batch(
            recs, drop, self.empty, self.device, self.rope, self.c.max_text_tokens
        )

    def forward_pair(self, batch, step: int, tstep=None, keep: str = "off"):
        """Teacher and student velocity on one noised batch; `keep` is the student's
        block-output mode; the teacher keeps detached outputs whenever the student
        keeps any."""
        p = (tstep or self.tstep).noise(batch, step)
        self.sb.clear()
        self.tb.clear()
        self.tb.mode = "off" if keep == "off" else "detach"
        with torch.no_grad(), self.m.autocast():
            v_t = self.m.teacher(p.z, p.text, p.sigma, p.seq)
        self.tb.mode, self.sb.mode = "off", keep
        with self.m.autocast():
            v_s = self.m.student(p.z, p.text, p.sigma, p.seq)
        self.sb.mode = "off"
        return p, v_s, v_t

    def set_anneal(self, t: float) -> None:
        self.t = t
        self.m.set_anneal(t)

    def band_norms(self, named) -> list[float]:
        acc = np.zeros(6)
        for name, g in named:
            if g is not None:
                parts = name.split(".")
                i = (
                    int(parts[1])
                    if parts[0] == "layers"
                    else (0 if parts[0] == "text_embedder" else self.depth - 1)
                )
                acc[band(i, self.depth)] += float(g.float().square().sum())
        return np.sqrt(acc).tolist()

    def _reduce(self, parts: dict[str, float]) -> dict[str, float]:
        for p in self.trained:
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad.div_(self.world)
        keys = sorted(parts)
        stat = torch.tensor([parts[x] for x in keys], device=self.device)
        dist.all_reduce(stat)
        return dict(zip(keys, (stat / self.world).tolist()))

    def train_loop(self) -> None:
        c = self.c
        rng = np.random.default_rng(c.seed + 1000 * self.rank)
        order, pos = rng.permutation(len(self.train)), 0
        log_file = (self.out / "metrics.jsonl").open("a") if self.rank == 0 else None
        keep = "keep" if self.loss.needs_blocks else "off"
        for step in range(c.steps):
            t0 = time.perf_counter()
            for o, bases in zip(self.opts, self.base_lr):
                for g, base in zip(o.param_groups, bases):
                    g["lr"] = base * lr_scale(step, c.steps, c.warmup)
            if c.anneal_steps > 0:
                self.set_anneal(anneal_t(step, c.anneal_steps))
            parts = {"soft": 0.0, "hard": 0.0, "conc": 0.0}
            for _ in range(c.grad_acc):
                if pos + c.micro > len(order):
                    order, pos = rng.permutation(len(self.train)), 0
                recs = [self.train[j] for j in order[pos : pos + c.micro]]
                pos += c.micro
                drop = (rng.random(c.micro) < c.text_drop).tolist()
                p, v_s, v_t = self.forward_pair(self.batch(recs, drop), step, keep=keep)
                loss, got = self.loss(v_s, v_t, p.target, self.sb.ordered())
                self.sb.clear()
                self.tb.clear()
                (loss / c.grad_acc).backward()
                for name, v in got.items():
                    parts[name] += float(v) / c.grad_acc
                del loss, v_s, v_t, p
            if self.world > 1:
                parts = self._reduce(parts)
            gain_bands = self.band_norms(
                (p_, ts[0].grad) for p_, ts in self.deltas.items()
            )
            trained_bands = self.band_norms(
                (n, p_.grad) for n, p_ in self.m.built.trainable.named_trainable()
            )
            writer_bands = self.band_norms((n, p_.grad) for n, p_ in self.writers)
            for o in self.opts:
                o.step()
                o.zero_grad(set_to_none=True)
            if log_file is not None:
                rec = {
                    "step": step,
                    "anneal_t": self.t,
                    "lr_scale": lr_scale(step, c.steps, c.warmup),
                    **{f"loss_{x}": v for x, v in parts.items()},
                }
                rec |= {
                    "grad_gain_bands": gain_bands,
                    "grad_adapter_bands": trained_bands,
                    "grad_writer_bands": writer_bands,
                    "secs": time.perf_counter() - t0,
                }
                log_file.write(json.dumps(rec) + "\n")
                if step % 25 == 0:
                    log_file.flush()
                    log.info(
                        f"step {step} soft {parts['soft']:.4f} hard {parts['hard']:.4f} conc {parts['conc']:.4f} t {self.t:.3f} {rec['secs']:.2f}s"
                    )
            self._checkpoints(step + 1)
        if log_file is not None:
            log_file.close()

    def _checkpoints(self, done: int) -> None:
        c = self.c
        evaluate_now = done % c.eval_every == 0 or done == c.steps
        snapshot = c.save_every and done % c.save_every == 0 and done < c.steps
        if self.rank == 0 and evaluate_now:
            self.record_eval(done)
        if self.rank == 0 and snapshot:
            save_parts(self, self.out / f"step_{done:05d}")
        if self.world > 1 and (evaluate_now or snapshot):
            dist.barrier()

    def record_eval(self, step: int) -> None:
        res = evaluate(self, step)
        with (self.out / "evals.jsonl").open("a") as f:
            f.write(json.dumps(res) + "\n")
        mid = res["sigma"]["0.5"]["v_rel_err_median"]
        log.info(
            f"eval {step}: v err (sigma 0.5) {mid:.4f}; "
            + " ".join(
                f"{s}:{v['v_rel_err_median']:.3f}" for s, v in res["sigma"].items()
            )
        )

    def free_optim(self) -> None:
        """Drop optimizer state and gradients (the end-of-run gradient map needs the room)."""
        self.opts.clear()
        for p in self.m.student.parameters():
            p.grad = None
        torch.cuda.empty_cache()


def run(env, c: HealConfig, k: dict, k_t: dict) -> None:
    """One rank of a heal run (`kohakumoss.parallel.launch.spawn` entry): build, the
    start eval and gradient map, train, the final parts, the end gradient map."""
    setup()
    out = Path(c.out)
    heal = Heal(c, k, k_t, env.device, env.rank, env.world)
    lead = env.rank == 0
    grads = {}
    if lead:
        if c.grad_map:
            grads["start"] = gradient_map(heal, "start")
        heal.record_eval(0)
    if env.world > 1:
        dist.barrier()
    heal.train_loop()
    if lead:
        save_parts(heal, out / "final")
        heal.free_optim()
        if c.grad_map:
            grads["end"] = gradient_map(heal, "end")
        (out / "grads.json").write_text(json.dumps(grads), encoding="utf-8")
    log.info(
        f"rank {env.rank}: peak memory {torch.cuda.max_memory_allocated(env.device) / 2**30:.2f} GiB"
    )
    if env.world > 1:
        dist.barrier()
