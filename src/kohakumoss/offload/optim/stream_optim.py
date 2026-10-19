"""GPU AdamW for streamed (`swap`) blocks, stepped per block inside backward of the
last micro-batch; states streamed or on the device. See docs/internals/offload.md."""

import weakref

import torch
from torch import Tensor

from kohakumoss.offload.optim.gpu_adamw import fused_adamw, make_groups
from kohakumoss.offload.weights.stream import WeightStream
from kohakumoss_common.calls import nothing
from kohakumoss_common.weak import weak_method

STATE_SLOTS = 3


def slot_bytes(
    block_numel: int, weight_bytes: int, device_states: bool, slots: int = STATE_SLOTS
) -> int:
    """Device bytes of StreamedAdamW's slot ring for blocks of `block_numel` streamed
    parameters: per slot an fp32 gradient and the streaming-dtype weight, plus fp32
    master / m / v rows for streamed states."""
    fp32_rows = 1 if device_states else 4
    return max(2, slots) * block_numel * (4 * fp32_rows + weight_bytes)


def _views(flat: Tensor, shapes: dict[str, torch.Size]) -> dict[str, Tensor]:
    """Consecutive views of `flat`, one per name, in `shapes` order."""
    out, offset = {}, 0
    for name, shape in shapes.items():
        n = shape.numel()
        out[name] = flat[offset : offset + n].view(shape)
        offset += n
    return out


class StreamedAdamW:
    """Per-block GPU AdamW over a trainable WeightStream. `groups`: torch-style groups
    over the stream's masters. `reduce(flat, block)` runs on each block's fp32 gradient
    sums before the transform (DDP: in-place SUM on the update stream, to the block's
    owner). Streamed states: only the owner (`weight_stream.mine`) holds a block's
    moments and steps it; device states: every rank steps every block, the owner
    writes the host master and streaming copy. `transform(g) -> found_inf | None` runs
    on each summed gradient before its update (unscale / clip; a nonzero found_inf
    skips it). Device bytes per block parameter: `state_slots` x (fp32 gradient +
    streaming weight), plus 3 fp32 rows (master, m, v) per slot for streamed states
    (`slot_bytes`)."""

    handles_decay_to_one = True

    def __init__(
        self,
        weight_stream: WeightStream,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        device_states: bool = False,
        state_slots: int = STATE_SLOTS,
        groups: list[dict] | None = None,
    ):
        ws = weight_stream
        if not ws.trainable:
            raise ValueError("StreamedAdamW needs a trainable (swap) WeightStream")
        self.weight_stream = ws
        self.device = ws.device
        self.streamed = ws.streamed
        self.dm = ws.dm
        self.device_states = device_states
        self.params = list(ws.params.values())
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.param_groups = make_groups(
            groups if groups is not None else self.params, defaults
        )
        by_id = {id(p): g for g in self.param_groups for p in g["params"]}
        missing = [k for k, p in ws.params.items() if id(p) not in by_id]
        if missing:
            raise ValueError(f"streamed parameters in no group: {missing[:4]}")
        self.group_of = {k: by_id[id(p)] for k, p in ws.params.items()}
        self.keys: list[list[tuple[int, str]]] = [[] for _ in range(ws.n)]
        for key in ws.params:
            self.keys[key[0]].append(key)
        pin = ws.pinned
        self.transform = nothing
        self.steps = {
            key: torch.zeros((), dtype=torch.float32, device=self.device)
            for key in ws.params
        }
        self.moments: dict[tuple[int, str], tuple[Tensor, Tensor]] = {}
        self.dev_state: dict[tuple[int, str], tuple[Tensor, Tensor, Tensor]] = {}
        self.state_keys = [k for k in ws.params if device_states or ws.mine(k[0])]
        for key in self.state_keys:
            master = ws.params[key]
            if device_states:
                m = master.detach().to(self.device)
                self.dev_state[key] = (m, torch.zeros_like(m), torch.zeros_like(m))
            else:
                self.moments[key] = tuple(
                    torch.zeros(master.shape, dtype=torch.float32, pin_memory=pin)
                    for _ in range(2)
                )
        names = [name for _, name in self.keys[0]] if self.keys else []
        shapes = {name: ws.layout[name][0] for name in names}
        self.n_slots = max(2, state_slots)
        # rows: master, m, v (streamed states only; device states live in dev_state)
        self.slots = [
            {
                name: torch.empty((3, *shape), dtype=torch.float32, device=self.device)
                for name, shape in shapes.items()
            }
            for _ in range(0 if device_states else self.n_slots)
        ]
        # the block's fp32 gradient sums, one flat buffer per slot (`reduce` sums it)
        total = sum(shape.numel() for shape in shapes.values())
        self.grad_flat = [
            torch.empty(total, dtype=torch.float32, device=self.device)
            for _ in range(self.n_slots)
        ]
        self.grad_views = [_views(flat, shapes) for flat in self.grad_flat]
        self.reduce = nothing
        # the updated weight in the streaming dtype, written back to the host copy
        self.weight_slots = [
            {
                name: torch.empty(shape, dtype=ws.layout[name][1], device=self.device)
                for name, shape in shapes.items()
            }
            for _ in range(self.n_slots)
        ]
        self.slot_of: dict[int, int] = {}
        self.next_slot = 0
        self.has_acc: dict[int, dict[str, bool]] = {}
        self.pending: dict[int, dict[str, Tensor]] = {}
        self.stats = {"bytes_h2d": 0, "bytes_d2h": 0}
        if self.streamed:
            cur = self.dm.current_stream(self.device)
            self.h2d = self.dm.Stream(self.device)
            self.d2h = self.dm.Stream(self.device)
            self.opt = self.dm.Stream(self.device)
            self.ready = [self.dm.Event() for _ in range(self.n_slots)]
            self.free = [self.dm.Event() for _ in range(self.n_slots)]
            for ev in self.free:
                ev.record(cur)
        if device_states:
            self._load = StreamedAdamW._load_device
            self._apply = StreamedAdamW._apply_device
        else:
            self._load = StreamedAdamW._load_streamed
            self._apply = StreamedAdamW._apply_streamed
        self._stepping = (weak_method(self._on_grad), weak_method(self._prefetch))
        ws.optimizer = weakref.ref(self)

    def start(self, last: bool) -> None:
        """Last micro-batch: step blocks in backward; else accumulate grads on host."""
        if last:
            on_grad, on_block_backward = self._stepping
            self.weight_stream.on_grad = on_grad
            self.weight_stream.on_block_backward = on_block_backward
        else:
            self.weight_stream.on_grad = self.weight_stream.accumulate
            self.weight_stream.on_block_backward = nothing

    def _prefetch(self, index: int) -> None:
        keys = self.keys[index]
        self.weight_stream.land_block(index, keys)
        filled = self.weight_stream.grad_filled
        self.has_acc[index] = {name: filled.get((i, name), False) for i, name in keys}
        self.pending[index] = {}
        self._load(self, index)

    def _take_slot(self, index: int) -> int:
        """The next ring slot for block `index`; raises while a block loaded into it
        has not been stepped (its master / moments / gradient rows are still live)."""
        slot = self.next_slot
        held = [i for i, s in self.slot_of.items() if s == slot]
        if held:
            raise RuntimeError(
                f"StreamedAdamW: block {index} needs state slot {slot}, still held by "
                f"block {held[0]} (not stepped yet; its last gradient has not arrived)"
            )
        self.next_slot = (slot + 1) % self.n_slots
        self.slot_of[index] = slot
        return slot

    def _load_device(self, index: int) -> None:
        slot = self._take_slot(index)
        if any(self.has_acc[index].values()):
            self._copy_in(index, slot, acc_only=True)

    def _load_streamed(self, index: int) -> None:
        acc_only = not self.weight_stream.mine(index)
        self._copy_in(index, self._take_slot(index), acc_only=acc_only)

    def _copy_in(self, index: int, slot: int, acc_only: bool) -> None:
        copies = []
        acc = self.has_acc[index]
        for key in self.keys[index]:
            name = key[1]
            master = self.weight_stream.params[key]
            if not acc_only:
                bufs = self.slots[slot]
                m, v = self.moments[key]
                copies += [
                    (bufs[name][0], master.data),
                    (bufs[name][1], m),
                    (bufs[name][2], v),
                ]
            if acc[name]:
                copies.append(
                    (self.grad_views[slot][name], self.weight_stream.host_grad[key])
                )
        if not self.streamed:
            for dst, src in copies:
                dst.copy_(src)
            return
        self.h2d.wait_event(self.free[slot])
        self.h2d.wait_event(self.weight_stream.host_ready[index])
        with self.dm.stream(self.h2d):
            for dst, src in copies:
                dst.copy_(src, non_blocking=True)
                self.stats["bytes_h2d"] += src.numel() * 4
        self.ready[slot].record(self.h2d)

    def _on_grad(self, key: tuple[int, str], grad: Tensor) -> None:
        index, name = key
        pending = self.pending[index]
        pending[name] = grad
        if len(pending) == len(self.keys[index]):
            self._step_block(index)

    def _update(self, index: int, slot: int, acc: dict, grads: dict) -> list:
        """This micro-batch's gradients + the earlier host sum (already in the slot,
        per key that has one) -> `reduce` -> the update (the blocks this rank steps)."""
        views = self.grad_views[slot]
        for _, name in self.keys[index]:
            if acc[name]:
                views[name].add_(grads[name])
            else:
                views[name].copy_(grads[name])
        self.reduce(self.grad_flat[slot], index)
        if not (self.device_states or self.weight_stream.mine(index)):
            return []
        outs = []
        for key in self.keys[index]:
            group = self.group_of[key]
            name = key[1]
            g = views[name]
            found = self.transform(g)
            weight = self.weight_slots[slot][name]
            step = self.steps[key]
            outs.append(self._apply(self, key, slot, g, step, group, found, weight))
        return outs

    @torch.no_grad()
    def _step_block(self, index: int) -> None:
        slot = self.slot_of.pop(index)
        acc = self.has_acc.pop(index)
        grads = self.pending.pop(index)
        if not self.streamed:
            self._write_back(index, slot, self._update(index, slot, acc, grads))
            return
        self.opt.wait_stream(self.dm.current_stream(self.device))
        self.opt.wait_event(self.ready[slot])
        self.opt.wait_event(self.free[slot])
        with self.dm.stream(self.opt):
            outs = self._update(index, slot, acc, grads)
        for g in grads.values():
            g.record_stream(self.opt)
        self._write_back(index, slot, outs)

    def _apply_streamed(self, key, slot, g, step, group, found, weight):
        buf = self.slots[slot][key[1]]
        master, m, v = buf[0], buf[1], buf[2]
        fused_adamw(master, g, m, v, step, group, found)
        weight.copy_(master)
        host_m, host_v = self.moments[key]
        pairs = [
            (self.weight_stream.params[key].data, master),
            (host_m, m),
            (host_v, v),
            (self.weight_stream.host[key[0]][key[1]], weight),
        ]
        return pairs, weight

    def _apply_device(self, key, slot, g, step, group, found, weight):
        master, m, v = self.dev_state[key]
        fused_adamw(master, g, m, v, step, group, found)
        if not self.weight_stream.mine(key[0]):
            return [], weight
        weight.copy_(master)
        pairs = [
            (self.weight_stream.params[key].data, master),
            (self.weight_stream.host[key[0]][key[1]], weight),
        ]
        return pairs, weight

    def _write_back(self, index: int, slot: int, outs) -> None:
        if not self.streamed:
            for pairs, _ in outs:
                for dst, src in pairs:
                    dst.copy_(src)
            return
        self.d2h.wait_stream(self.opt)
        with self.dm.stream(self.d2h):
            for pairs, weight in outs:
                for dst, src in pairs:
                    dst.copy_(src, non_blocking=True)
                    self.stats["bytes_d2h"] += src.nbytes
        self.free[slot].record(self.d2h)
        self.weight_stream.host_ready[index].record(self.d2h)

    def finish(self) -> None:
        """After the last micro-batch's backward: the compute stream waits for the block
        updates and the host for the state copies; the step's host gradient sums end
        (`WeightStream.zero_grads`) and the ring forgets every slot, so the next forward
        reloads the updated weights; raises when a block loaded for its step was never
        stepped."""
        if self.slot_of:
            raise RuntimeError(
                f"StreamedAdamW: blocks {sorted(self.slot_of)} were loaded in backward "
                "but never stepped (a gradient of theirs did not arrive)"
            )
        if self.streamed:
            self.dm.current_stream(self.device).wait_stream(self.opt)
            self.h2d.synchronize()
        self.weight_stream.zero_grads()
        self.weight_stream.ring.forget()

    def synchronize(self) -> None:
        """Wait until every write-back has landed in host memory."""
        if self.streamed:
            self.d2h.synchronize()

    def masters(self) -> dict[tuple[int, str], Tensor]:
        """Current fp32 masters (the host parameters)."""
        self.synchronize()
        return {k: p.detach() for k, p in self.weight_stream.params.items()}

    def zero_grad(self, set_to_none: bool = True) -> None:
        return None

    def state_dict(self) -> dict:
        """Moments and step per stream key this rank holds (masters are the
        parameters)."""
        self.synchronize()
        out = {}
        for key in self.state_keys:
            m, v = self.dev_state[key][1:] if self.device_states else self.moments[key]
            out[f"{key[0]}:{key[1]}"] = (m, v, self.steps[key].clone())
        return out

    def load_state_dict(self, state: dict) -> None:
        """Restore moments / steps; masters come from the parameters (re-pushed to the
        device-state copies and the streaming weights)."""
        for key in self.state_keys:
            master = self.weight_stream.params[key]
            m, v, step = state[f"{key[0]}:{key[1]}"]
            dst = (
                self.dev_state[key]
                if self.device_states
                else (None, *self.moments[key])
            )
            dst[1].copy_(m)
            dst[2].copy_(v)
            self.steps[key].fill_(float(step))
            if self.device_states:
                dst[0].copy_(master.data)
        self.weight_stream.refresh()


__all__ = ["StreamedAdamW"]
