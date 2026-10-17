"""Multi-process self-checks (importable by spawned workers): each worker compares its
distributed result against a single-process reference it computes itself and writes
the max error into a shared result file."""

import json
import os
import time

import torch
import torch.distributed as dist

from kohakumoss.parallel.ddp import GradReducer
from kohakumoss.parallel.fsdp import shard
from kohakumoss.parallel.pipeline import (
    SINGLE,
    MicroBatch,
    PipelineEngine,
    PipeShape,
)
from kohakumoss.parallel.pipeline import microbatch_to as _to
from kohakumoss.training.memory.synthetic import SyntheticModel
from kohakumoss.types.seqinfo import build_seqinfo

DIM, FFN, DEPTH, CHANS = 32, 64, 4, 8


def _model(seed: int = 0, dtype=torch.float64) -> SyntheticModel:
    torch.manual_seed(seed)
    return SyntheticModel(DIM, FFN, DEPTH, CHANS).to(dtype)


def _micro(seed: int, hw, text, dtype=torch.float64, keep=None) -> MicroBatch:
    """`keep[i]`: kept grid indices of sample i (token drop), or None for all."""
    g = torch.Generator().manual_seed(seed)
    keep_t = (
        None if keep is None else [None if k is None else torch.tensor(k) for k in keep]
    )
    keeps = keep_t or [None] * len(hw)
    n_img = sum(h * w if k is None else k.numel() for (h, w), k in zip(hw, keeps))
    return MicroBatch(
        img=torch.randn(n_img, CHANS, generator=g, dtype=dtype),
        text=torch.randn(sum(text), CHANS, generator=g, dtype=dtype),
        sigma=torch.rand(len(hw), generator=g, dtype=torch.float32),
        hw=torch.tensor(hw),
        text_len=torch.tensor(text),
        target=torch.randn(n_img, CHANS, generator=g, dtype=dtype),
        weight=torch.ones(len(hw)),
        img_keep=keep_t,
    )


def _mse(pred, mb: MicroBatch):
    return (pred.double() - mb.target.double()).square().mean(-1).sum()


def _write(path: str, rank: int, payload: dict) -> None:
    with open(f"{path}.{rank}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)


def ddp_worker(env, result_path: str) -> None:
    """Each rank backprops its own batch; reduced grads must equal the sum over ranks."""
    model = _model().to(env.device)
    reducer = GradReducer(model.parameters())
    batches = [
        _to(_micro(100 + r, [(2, 3), (1, 4)], [3, 2]), env.device)
        for r in range(env.world)
    ]
    denom = float(sum(b.target.shape[0] for b in batches))
    reducer.arm(True)
    pred = _forward(model, batches[env.rank])
    (_mse(pred, batches[env.rank]) / denom).backward()
    reducer.finish()
    ref = _model().to(env.device)
    for b in batches:
        (_mse(_forward(ref, b), b) / denom).backward()
    err = max(
        (p.grad - q.grad).abs().max().item()
        for p, q in zip(model.parameters(), ref.parameters())
    )
    _write(result_path, env.rank, {"max_err": err})


def _forward(model: SyntheticModel, mb: MicroBatch):
    seq = build_seqinfo(mb.hw, mb.text_len, img_keep=mb.img_keep).to(mb.img.device)
    return model(mb.img, mb.text, mb.sigma, seq)


def pipeline_worker(env, result_path: str, schedule: str) -> None:
    """Two-stage pipeline grads/loss must equal the single-process reference."""
    model = _model()
    micro = [
        _micro(1, [(2, 3), (1, 4)], [3, 2]),
        _micro(2, [(3, 3), (4, 5)], [4, 2], keep=[None, [0, 3, 7, 12, 19]]),
    ]
    bounds, per_rank = [(0, 2), (2, 4)], 1
    if schedule not in SINGLE:
        micro += [_micro(3, [(1, 5)], [2]), _micro(4, [(2, 2), (2, 2)], [1, 1])]
        bounds, per_rank = [(i, i + 1) for i in range(DEPTH)], 2
    denom = float(sum(m.target.shape[0] for m in micro))
    shape = PipeShape(tokens=32, max_samples=2, dim=DIM, microbatches=len(micro))
    engine = PipelineEngine(
        model,
        env.rank,
        env.world,
        env.device,
        shape,
        bounds,
        schedule,
        _mse,
        torch.float64,
        per_rank=per_rank,
    )
    loss = engine.step(micro, denom)
    ref = _model()
    ref_loss = sum(_mse(_forward(ref, m), m) / denom for m in micro)
    ref_loss.backward()
    ref_grads = dict(ref.named_parameters())
    errs = []
    for mod in engine.modules_:
        for name, p in mod.named_parameters():
            if p.grad is None:
                continue
            key = _ref_name(name, mod)
            errs.append((p.grad.cpu() - ref_grads[key].grad).abs().max().item())
    payload = {"max_err": max(errs), "n": len(errs)}
    if loss is not None:
        payload["loss_err"] = abs(loss.item() - ref_loss.item()) / abs(ref_loss.item())
    _write(result_path, env.rank, payload)


def _ref_name(name: str, mod) -> str:
    if name.startswith("layers."):
        return name
    if name.startswith("head_parts."):
        idx, rest = name[len("head_parts.") :].split(".", 1)
        return ["embed", "text", "time"][int(idx)] + "." + rest
    if name.startswith("tail."):
        _, rest = name[len("tail.") :].split(".", 1)
        return "out." + rest
    raise KeyError(name)


def fsdp_worker(env, result_path: str) -> None:
    """FSDP2-sharded grads must equal the unsharded reference (CUDA, nccl)."""
    model = _model(dtype=torch.float32).to(env.device)
    shard(model, model.layers)
    batches = [
        _micro(200 + r, [(2, 3), (1, 4)], [3, 2], torch.float32)
        for r in range(env.world)
    ]
    denom = float(sum(b.target.shape[0] for b in batches))
    mb = _to(batches[env.rank], env.device)
    (_mse(_forward(model, mb), mb) / denom).backward()
    ref = _model(dtype=torch.float32).to(env.device)
    for b in batches:
        b = _to(b, env.device)
        (_mse(_forward(ref, b), b) / denom).backward()
    errs = []
    for (name, p), q in zip(model.named_parameters(), ref.parameters()):
        g = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
        errs.append((g - q.grad).abs().max().item())
    _write(result_path, env.rank, {"max_err": max(errs)})


def comm_worker(env, result_path: str, sizes_mb: tuple[int, ...] = (1, 16, 64)) -> None:
    """all_reduce bandwidth (algorithm GB/s) per message size."""
    out = {}
    for mb in sizes_mb:
        t = torch.ones(mb * 1024 * 1024 // 4, device=env.device)
        for _ in range(3):
            dist.all_reduce(t)
        _sync(env.device)
        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            dist.all_reduce(t)
        _sync(env.device)
        dt = (time.perf_counter() - t0) / iters
        out[str(mb)] = {"ms": dt * 1e3, "algo_GBps": mb / 1024 / dt}
    _write(result_path, env.rank, out)


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def read_results(path: str, world: int) -> list[dict]:
    out = []
    for r in range(world):
        with open(f"{path}.{r}.json", encoding="utf-8") as f:
            out.append(json.load(f))
        os.remove(f"{path}.{r}.json")
    return out


__all__ = [
    "comm_worker",
    "ddp_worker",
    "fsdp_worker",
    "pipeline_worker",
    "read_results",
]
