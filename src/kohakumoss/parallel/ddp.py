"""Data parallel: bucketed SUM all-reduce of the trainable gradients that overlaps
backward (gradients accumulate in place in the bucket buffers), the side-stream
reduction of gradients that leave backward early (in-backward steps, gradient offload,
streamed blocks), or torch DDP. See docs/guides/multi-gpu.md."""

import contextlib
from collections.abc import Callable
from functools import partial

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.nn.parallel import DistributedDataParallel

from kohakumoss.offload.accel import module, streamed
from kohakumoss.parallel.telemetry.timing import build_clock
from kohakumoss_common.runs import runs_within
from kohakumoss_common.units import MIB
from kohakumoss_common.weak import weak_call, weak_method

WIDE_AHEAD = 2  # reduce-dtype bucket copies in flight before the oldest one lands


class _Bucket:
    """Same-dtype, same-device parameters whose gradients live in one flat buffer;
    with a `reduce_dtype` other than theirs the reduction runs on a copy in that dtype
    (made at launch, copied back once it lands)."""

    def __init__(
        self, params: list[nn.Parameter], reduce_dtype: torch.dtype | None = None
    ):
        self.params = params
        p0 = params[0]
        self.flat = torch.zeros(
            sum(p.numel() for p in params), dtype=p0.dtype, device=p0.device
        )
        wide = reduce_dtype is not None and reduce_dtype != p0.dtype
        self.reduce_dtype = reduce_dtype if wide else None
        self.wide: Tensor | None = None
        self.views = []
        offset = 0
        for p in params:
            n = p.numel()
            self.views.append(self.flat[offset : offset + n].view_as(p))
            offset += n
        self.pending = len(params)
        self.work = None

    def reduced_buffer(self) -> Tensor:
        """The buffer the all-reduce sums: the flat buffer, or its copy in the
        reduce dtype."""
        if self.reduce_dtype is None:
            return self.flat
        self.wide = self.flat.to(self.reduce_dtype)
        return self.wide

    def copy_back(self) -> None:
        """After the reduction: the summed copy back into the gradients' buffer."""
        if self.wide is not None:
            self.flat.copy_(self.wide)
            self.wide = None

    @property
    def reduced_bytes(self) -> int:
        dtype = self.reduce_dtype or self.flat.dtype
        return self.flat.numel() * torch.empty((), dtype=dtype).element_size()

    def attach(self) -> None:
        """Zero the buffer and make it every parameter's gradient (the step's first
        micro-batch)."""
        self.flat.zero_()
        for p, view in zip(self.params, self.views):
            p.grad = view

    def gather(self) -> None:
        """Copy in a gradient that something replaced with its own tensor."""
        for p, view in zip(self.params, self.views):
            if p.grad is None:
                view.zero_()
                p.grad = view
            elif p.grad.data_ptr() != view.data_ptr():
                view.copy_(p.grad)
                p.grad = view


def _skip(reducer, bucket: _Bucket, p: Tensor) -> None:
    return None


class GradReducer:
    """Bucketed SUM all-reduce of `params`' gradients, launched from the post-accumulate
    hooks of the step's last micro-batch so it overlaps the rest of backward.

    Gradients are views into the bucket buffers (no copy in or out); a bucket of a
    dtype other than `reduce_dtype` (default fp32) is summed on a copy in it (None:
    in the parameters' dtype), and at most WIDE_AHEAD such copies are alive: launching
    one more lands the oldest. `finish` makes the compute stream wait for the
    reductions and its device time is the exposed communication (`metrics`)."""

    def __init__(
        self,
        params,
        group=None,
        bucket_bytes: int = 25 * MIB,
        average: bool = False,
        reduce_dtype: torch.dtype | None = torch.float32,
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
                self.buckets.append(_Bucket(current, reduce_dtype))
                current, size = [], 0
            current.append(p)
            size += p.nbytes
        if current:
            self.buckets.append(_Bucket(current, reduce_dtype))
        self.bytes = sum(b.reduced_bytes for b in self.buckets)
        device = self.params[0].device if self.params else torch.device("cpu")
        self.clock = build_clock(device)
        self._active = _skip
        self.next = 0
        self.widened: list[_Bucket] = []
        self.handles = [
            p.register_post_accumulate_grad_hook(weak_call(GradReducer._hook, self, i))
            for i, bucket in enumerate(self.buckets)
            for p in bucket.params
        ]

    def start(self, last_microbatch: bool) -> None:
        """Reduce during this backward (last micro-batch) or only accumulate; the
        step's first call attaches the bucket buffers as the gradients."""
        self._active = GradReducer._count if last_microbatch else _skip
        for b in self.buckets:
            if b.params[0].grad is None:
                b.attach()
            b.pending = len(b.params)
            b.work = None
        self.next = 0
        self.widened = []
        if last_microbatch:
            self.clock.begin()

    def _hook(self, index: int, p: Tensor) -> None:
        self._active(self, self.buckets[index], p)

    def _count(self, bucket: _Bucket, p: Tensor) -> None:
        bucket.pending -= 1
        if bucket.pending == 0:
            self._launch_ready()

    def _launch_ready(self) -> None:
        """Launch the complete buckets in bucket order (the same on every rank)."""
        while self.next < len(self.buckets) and self.buckets[self.next].pending == 0:
            self._launch(self.buckets[self.next])
            self.next += 1

    def _launch(self, bucket: _Bucket) -> None:
        bucket.gather()
        buf = bucket.reduced_buffer()
        bucket.work = dist.all_reduce(buf, group=self.group, async_op=True)
        if bucket.reduce_dtype is not None:
            self.widened.append(bucket)
            while len(self.widened) > WIDE_AHEAD:
                self._wait_bucket(self.widened.pop(0))

    @torch.no_grad()
    def _wait_bucket(self, bucket: _Bucket) -> None:
        """The current stream waits for `bucket`'s reduction; its sum lands in the
        gradients (the reduce-dtype copy is released)."""
        if bucket.work is None:
            return
        bucket.work.wait()
        bucket.work = None
        bucket.copy_back()
        if self.average:
            bucket.flat.div_(self.world)

    @torch.no_grad()
    def wait(self) -> None:
        """Make the current stream wait for every bucket's reduction (launching, in
        order, the buckets whose hooks did not all fire)."""
        for bucket in self.buckets[self.next :]:
            self._launch(bucket)
        self.next = len(self.buckets)
        for bucket in self.buckets:
            self._wait_bucket(bucket)
        self.widened = []

    def finish(self) -> None:
        """`wait`, timed as the step's exposed communication."""
        mark = self.clock.start()
        self.wait()
        self.clock.stop("comm_wait", mark)
        self.clock.end()

    def metrics(self) -> dict[str, float]:
        """The last step's exposed reduction time (`ddp_comm_wait_ms`), its span from
        the last micro-batch's start to the end of the wait (`ddp_span_ms`) and the
        reduced megabytes per step (`ddp_reduce_mb`)."""
        t = self.clock.read()
        out = {"ddp_reduce_mb": self.bytes / MIB}
        if t:
            out["ddp_comm_wait_ms"] = t.get("comm_wait_ms", 0.0)
            out["ddp_span_ms"] = t["wall_ms"]
        return out

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


def plan_buckets(items: list[tuple[int, object]], bucket_bytes: int) -> list[list[int]]:
    """Indices of `items` ((bytes, key) in registration order) grouped into buckets:
    reverse order, one key (dtype) per bucket, a new bucket once one holds
    `bucket_bytes`."""
    out, current, size = [], [], 0
    for i in reversed(range(len(items))):
        nbytes, key = items[i]
        if current and (size >= bucket_bytes or items[current[0]][1] != key):
            out.append(current)
            current, size = [], 0
        current.append(i)
        size += nbytes
    if current:
        out.append(current)
    return out


class _Slab:
    """One static bucket of an AsyncReducer: its keys' flat offsets and deliveries, the
    flat buffer while a step fills it (the gradients, then one presence count per key),
    the keys that arrived."""

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        self.keys: list[int] = []
        self.place: dict[int, tuple[int, torch.Size]] = {}
        self.index: dict[int, int] = {}
        self.deliver: dict[int, Callable[[Tensor], None]] = {}
        self.numel = 0
        self.flat: Tensor | None = None
        self.arrived: set[int] = set()
        self.done = False

    def add_key(self, key: int, shape, deliver: Callable[[Tensor], None]) -> None:
        shape = torch.Size(shape)
        self.place[key] = (self.numel, shape)
        self.index[key] = len(self.keys)
        self.deliver[key] = deliver
        self.keys.append(key)
        self.numel += shape.numel()

    @property
    def size(self) -> int:
        """Flat elements: the gradients and the presence counts."""
        return self.numel + len(self.keys)

    def view(self, key: int) -> Tensor:
        offset, shape = self.place[key]
        return self.flat[offset : offset + shape.numel()].view(shape)

    def presence(self, key: int) -> Tensor:
        """Key's presence count (1 where it arrived; the SUM of the ranks' after the
        reduction)."""
        at = self.numel + self.index[key]
        return self.flat[at : at + 1]

    def counts(self) -> Tensor:
        return self.flat[self.numel :]

    def reset(self) -> None:
        self.flat = None
        self.arrived.clear()
        self.done = False


def _deliver(p: nn.Parameter, then: Callable, grad: Tensor) -> None:
    p.grad = grad if grad.dtype == p.dtype else grad.to(p.dtype)
    then(p)


class AsyncReducer:
    """SUM all-reduce of gradients that leave backward before the optimiser step, on a
    side stream so it overlaps the rest of backward.

    `reduced(params, then)` gives a per-parameter hook body: on the step's last
    micro-batch each `p.grad` is copied into its static bucket (reverse order, one dtype,
    `bucket_bytes`); complete buckets are reduced asynchronously in bucket order and each
    `then(p)` runs on the side stream with the summed gradient as `p.grad`.
    `reduce_flat(t)` sums a caller's buffer on the caller's stream (streamed blocks).
    `finish` launches the buckets still open (a gradient absent on a rank counts as
    zero there; one absent on every rank is not delivered, as on one GPU) and makes
    the compute stream wait for the side stream, every `watch`ed stream and the plain
    `inner` GradReducer; that wait is `ddp_comm_wait_ms`. Off an accelerator every
    reduction is synchronous."""

    def __init__(
        self,
        device: torch.device | str,
        group=None,
        bucket_bytes: int = 25 * MIB,
        inner: GradReducer | None = None,
        reduce_dtype: torch.dtype | None = torch.float32,
    ):
        self.device = torch.device(device)
        self.group = group
        self.bucket_bytes = bucket_bytes
        self.inner = inner
        self.reduce_dtype = reduce_dtype
        self.side = streamed(self.device)
        self.dm = module(self.device) if self.side else None
        self.stream = self.dm.Stream(self.device) if self.side else None
        self.clock = build_clock(self.device)
        self.slabs: list[_Slab] = []
        self.slab_of: dict[int, _Slab] = {}
        self.watched: list = []
        self.on_start: list[Callable[[bool], None]] = []
        self.next = 0
        self.host_reads = False
        self.step_bytes = 0
        self.last_bytes = 0

    def reduced(self, params, then: Callable[[nn.Parameter], None]) -> Callable:
        """Track `params` in static buckets; returns the hook body `(p) -> None`
        handing `p.grad` to the reduction, `then(p)` running once it is summed (in
        `reduce_dtype`, default fp32; None: the parameters' dtype), as a gradient of
        the parameter's dtype. The hook body holds this reducer weakly."""
        params = [p for p in params if p.requires_grad]
        items = [(p.nbytes, p.dtype) for p in params]
        for idx in plan_buckets(items, self.bucket_bytes):
            slab = _Slab(self.reduce_dtype or params[idx[0]].dtype)
            for i in idx:
                p = params[i]
                slab.add_key(id(p), p.shape, partial(_deliver, p, then))
                self.slab_of[id(p)] = slab
            self.slabs.append(slab)
        return weak_method(self.add_param)

    def watch(self, stream) -> None:
        """A stream whose reductions the compute stream waits for at `finish`."""
        if stream is not None:
            self.watched.append(stream)

    def _on_side(self):
        return self.dm.stream(self.stream) if self.side else contextlib.nullcontext()

    def start(self, last: bool) -> None:
        """Before each micro-batch: the last one opens the step's buckets and clock."""
        if self.inner is not None:
            self.inner.start(last)
        for fn in self.on_start:
            fn(last)
        if last:
            for slab in self.slabs:
                slab.reset()
            self.next = 0
            self.step_bytes = 0
            self.clock.begin()

    def add_param(self, p: nn.Parameter) -> None:
        self.add(id(p), p.grad)

    @torch.no_grad()
    def add(self, key: int, grad: Tensor) -> None:
        """Copy `grad` into its bucket on the side stream (after the current stream's
        work); the bucket is reduced and delivered once every key of it arrived."""
        slab = self.slab_of[key]
        if slab.done:
            raise RuntimeError("a gradient arrived for a bucket already reduced")
        if self.side:
            self.stream.wait_stream(self.dm.current_stream(self.device))
        with self._on_side():
            if slab.flat is None:
                slab.flat = torch.empty(slab.size, dtype=slab.dtype, device=self.device)
            slab.view(key).copy_(grad)
            slab.presence(key).fill_(1)
        if self.side:
            grad.record_stream(self.stream)
        slab.arrived.add(key)
        if len(slab.arrived) == len(slab.keys):
            self._launch_ready()

    def _launch_ready(self) -> None:
        """Launch the complete buckets in bucket order (the same on every rank)."""
        while self.next < len(self.slabs):
            slab = self.slabs[self.next]
            if len(slab.arrived) < len(slab.keys):
                return
            self._launch(slab)
            self.next += 1

    def _launch(self, slab: _Slab) -> None:
        """Reduce `slab` and deliver its keys; with a key missing here, the summed
        presence counts are read (one host sync) and a key no rank sent is skipped."""
        absent = [key for key in slab.keys if key not in slab.arrived]
        with self._on_side():
            if slab.flat is None:
                slab.flat = torch.zeros(slab.size, dtype=slab.dtype, device=self.device)
            for key in absent:
                slab.view(key).zero_()
                slab.presence(key).zero_()
            self.reduce_flat(slab.flat)
            counts = slab.counts().tolist() if absent else None
            for key in slab.keys:
                if counts is None or counts[slab.index[key]] > 0:
                    slab.deliver[key](slab.view(key))
        slab.flat = None
        slab.done = True

    def reduce_flat(self, flat: Tensor) -> None:
        """In-place SUM all-reduce of `flat`; the current stream waits for it (on an
        accelerator the host does not)."""
        self.step_bytes += flat.nbytes
        work = dist.all_reduce(flat, group=self.group, async_op=self.side)
        if self.side:
            work.wait()

    def block_reduce(self, owners: list[int], everyone: bool = False) -> Callable:
        """`reduce(flat, block)` for streamed blocks: in-place SUM onto `owners[block]`
        (the rank whose host optimiser shard steps it), or onto every rank
        (`everyone`); the current stream waits for it."""
        return partial(self._reduce_block, owners, everyone)

    def _reduce_block(self, owners, everyone: bool, flat: Tensor, index: int) -> None:
        if everyone:
            self.reduce_flat(flat)
            return
        self.step_bytes += flat.nbytes
        work = dist.reduce(flat, owners[index], group=self.group, async_op=self.side)
        if self.side:
            work.wait()

    @torch.no_grad()
    def finish(self) -> None:
        """Before the optimiser step: every reduction and delivery has landed for the
        compute stream (and for the host when `host_reads`)."""
        mark = self.clock.start()
        if self.inner is not None:
            self.inner.wait()
        for slab in self.slabs[self.next :]:
            self._launch(slab)
        self.next = len(self.slabs)
        if self.side:
            cur = self.dm.current_stream(self.device)
            for stream in (self.stream, *self.watched):
                cur.wait_stream(stream)
            if self.host_reads:
                self.stream.synchronize()
        self.clock.stop("comm_wait", mark)
        self.clock.end()
        inner = self.inner.bytes if self.inner is not None else 0
        self.last_bytes = self.step_bytes + inner

    def metrics(self) -> dict[str, float]:
        """The last step's exposed reduction wait (`ddp_comm_wait_ms`), its span from
        the last micro-batch's start (`ddp_span_ms`) and the reduced megabytes
        (`ddp_reduce_mb`)."""
        t = self.clock.read()
        out = {"ddp_reduce_mb": self.last_bytes / MIB}
        if t:
            out["ddp_comm_wait_ms"] = t.get("comm_wait_ms", 0.0)
            out["ddp_span_ms"] = t["wall_ms"]
        return out


def _square_norm(grads: list[Tensor], device: torch.device) -> Tensor:
    if not grads:
        return torch.zeros((), dtype=torch.float32, device=device)
    norms = torch._foreach_norm(grads, 2.0, dtype=torch.float32)
    return torch.stack([n.to(device) for n in norms]).square().sum()


class ShardedNorm:
    """Global fp32 L2 gradient norm when the gradients `sharded()` names (by id) live
    on one rank each (host optimiser shards) and every other one is replicated."""

    def __init__(self, sharded: Callable[[], set[int]], device, group=None):
        self.sharded = sharded
        self.device = torch.device(device)
        self.group = group

    def __call__(self, grads: list[Tensor]) -> Tensor:
        ids = self.sharded()
        own = [g for g in grads if id(g) in ids]
        rest = [g for g in grads if id(g) not in ids]
        sq = _square_norm(own, self.device).reshape(1)
        dist.all_reduce(sq, group=self.group)
        return (sq[0] + _square_norm(rest, self.device)).sqrt()


def sum_block_stats(bad: Tensor, sq: Tensor, group=None) -> None:
    """In-place SUM over the ranks of the in-backward block steps' non-finite count
    and squared gradient norm (each rank stepped the blocks it owns), in one
    all-reduce."""
    both = torch.cat([bad.reshape(-1).float(), sq.reshape(-1).float()])
    dist.all_reduce(both, group=group)
    bad.copy_(both[: bad.numel()].reshape(bad.shape))
    sq.copy_(both[bad.numel() :].reshape(sq.shape))


def _sum_hook(state: tuple, bucket):
    """torch DDP comm hook: SUM all-reduce of the bucket (on a copy in `reduce_dtype`
    when the bucket has another dtype); `state` is (group, reduce_dtype)."""
    group, reduce_dtype = state
    buf = bucket.buffer()
    wide = buf if reduce_dtype in (None, buf.dtype) else buf.to(reduce_dtype)
    fut = dist.all_reduce(wide, group=group, async_op=True).get_future()

    def copy_back(done) -> Tensor:
        if wide is not buf:
            buf.copy_(wide)
        return buf

    return fut.then(copy_back)


def wrap_torch_ddp(
    model: nn.Module,
    device: torch.device,
    reduce_dtype: torch.dtype | None = torch.float32,
    group=None,
    **kwargs,
) -> nn.Module:
    """Stock torch DDP over `model` (DDP_BACKEND='torch', the A / B baseline): no
    buffer broadcast per forward, gradients a SUM over ranks in `reduce_dtype` (None:
    the bucket's dtype), as the project's reducers sum."""
    ids = [device.index] if device.type == "cuda" else None
    ddp = DistributedDataParallel(
        model, device_ids=ids, broadcast_buffers=False, process_group=group, **kwargs
    )
    ddp.register_comm_hook((group, reduce_dtype), _sum_hook)
    return ddp


def _chunks(tensors: list[Tensor], limit: int | None) -> list[list[Tensor]]:
    """Consecutive runs of `tensors` of at most `limit` bytes (one run when None)."""
    return runs_within(tensors, _nbytes, limit)


def _nbytes(t: Tensor) -> int:
    return t.nbytes


@torch.no_grad()
def broadcast_params(
    params, src: int = 0, group=None, device=None, chunk_bytes: int = 256 * MIB
) -> None:
    """Make every rank start from rank `src`'s values (adapter init): one broadcast
    per (device, dtype) group of flattened tensors. Host tensors (streamed masters) go
    through accelerator `device`, when given, in runs of at most `chunk_bytes`."""
    groups: dict[tuple, list[Tensor]] = {}
    for p in params:
        groups.setdefault((p.device, p.dtype), []).append(p.data)
    for (where, _), tensors in groups.items():
        stage = device if device is not None and where.type == "cpu" else None
        stage = stage if stage is not None and streamed(stage) else None
        for part in _chunks(tensors, chunk_bytes if stage is not None else None):
            flat = _flatten_dense_tensors(part)
            sent = flat if stage is None else flat.to(stage)
            dist.broadcast(sent, src=src, group=group)
            if stage is not None:
                flat.copy_(sent)
            for t, v in zip(part, _unflatten_dense_tensors(flat, part)):
                t.copy_(v)
