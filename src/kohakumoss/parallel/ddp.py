"""Data parallel: bucketed SUM all-reduce of the trainable gradients that overlaps
backward, or torch DDP. See docs/guides/multi-gpu.md."""

from functools import partial

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel


class _Bucket:
    def __init__(self, params: list[nn.Parameter]):
        self.params = params
        numel = sum(p.numel() for p in params)
        p0 = params[0]
        self.flat = torch.empty(numel, dtype=p0.dtype, device=p0.device)
        self.views = []
        offset = 0
        for p in params:
            n = p.numel()
            self.views.append(self.flat[offset : offset + n].view_as(p))
            offset += n
        self.pending = len(params)
        self.work = None


def _skip(bucket: _Bucket, p: Tensor) -> None:
    return None


class GradReducer:
    """Bucketed SUM all-reduce of `params`' gradients, overlapped with backward."""

    def __init__(
        self,
        params,
        group=None,
        bucket_bytes: int = 25 * 1024 * 1024,
        average: bool = False,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.group = group
        self.world = dist.get_world_size(group)
        self.average = average
        self.buckets: list[_Bucket] = []
        current, size = [], 0
        for p in reversed(self.params):
            key = (p.dtype, p.device)
            if current and (
                size >= bucket_bytes or (current[0].dtype, current[0].device) != key
            ):
                self.buckets.append(_Bucket(current))
                current, size = [], 0
            current.append(p)
            size += p.numel() * p.element_size()
        if current:
            self.buckets.append(_Bucket(current))
        self._active = _skip
        self.handles = [
            p.register_post_accumulate_grad_hook(partial(self._hook, bucket))
            for bucket in self.buckets
            for p in bucket.params
        ]

    def arm(self, last_microbatch: bool) -> None:
        """Reduce during this backward (last micro-batch) or only accumulate."""
        self._active = self._count if last_microbatch else _skip
        for b in self.buckets:
            b.pending = len(b.params)
            b.work = None

    def _hook(self, bucket: _Bucket, p: Tensor) -> None:
        self._active(bucket, p)

    def _count(self, bucket: _Bucket, p: Tensor) -> None:
        bucket.pending -= 1
        if bucket.pending == 0:
            self._launch(bucket)

    def _launch(self, bucket: _Bucket) -> None:
        for p, view in zip(bucket.params, bucket.views):
            if p.grad is None:
                view.zero_()
            else:
                view.copy_(p.grad)
        bucket.work = dist.all_reduce(bucket.flat, group=self.group, async_op=True)

    @torch.no_grad()
    def finish(self) -> None:
        """Wait for every bucket and write the reduced gradients back."""
        for bucket in self.buckets:
            if bucket.work is None:
                self._launch(bucket)
            bucket.work.wait()
            if self.average:
                bucket.flat.div_(self.world)
            for p, view in zip(bucket.params, bucket.views):
                if p.grad is None:
                    p.grad = view.clone()
                else:
                    p.grad.copy_(view)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


def wrap_torch_ddp(model: nn.Module, device: torch.device, **kwargs) -> nn.Module:
    """Stock DDP (for comparison / full FT without FSDP)."""
    ids = [device.index] if device.type == "cuda" else None
    return DistributedDataParallel(model, device_ids=ids, **kwargs)


def broadcast_params(params, src: int = 0, group=None) -> None:
    """Make every rank start from rank `src`'s values (adapter init)."""
    with torch.no_grad():
        for p in params:
            dist.broadcast(p.data, src=src, group=group)


def all_reduce_sum(t: Tensor, group=None) -> Tensor:
    """In-place SUM all-reduce (token counts, loss sums)."""
    dist.all_reduce(t, group=group)
    return t
