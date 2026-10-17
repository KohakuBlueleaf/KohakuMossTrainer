"""The two fused kits a layer binds from: `Pinned(backend)` calls one backend's
kernels directly, `Raced(backend)` binds each op to a `race.TierSlot`. Both expose the
same factories; each returns the callable for a `bind.fused` shell, or None."""

from functools import partial

from lycoris.functional.diag_oft import MAX_BLOCK
from lycoris.functional.lokr import apply_supported
from lycoris.kernels import MAX_HADA_RANK

from kohakumoss.adapters.bind import fused, ops
from kohakumoss.adapters.bind.race import TierSlot, flat, static_bucket, token_bucket


class Pinned:
    def __init__(self, backend: str):
        self.backend = backend

    def lora(self, lyco, gamma):
        return partial(fused.lora_diff, lyco, gamma, self.backend)

    def loha(self, lyco, gamma):
        if ops.hada_rank(lyco) > MAX_HADA_RANK:
            return None
        return partial(fused.loha_diff, lyco, gamma, self.backend)

    def lokr(self, lyco, scale):
        raw = fused.kron_factors(lyco, fused.kron_getters(lyco, False))
        if not apply_supported(*raw, None):
            return None
        return partial(
            fused.lokr_diff, lyco, fused.kron_getters(lyco), scale, self.backend
        )

    def oft(self, linear, lyco):
        if lyco.block_size > MAX_BLOCK:
            return None
        fn = fused.oft_bypass_rescaled if lyco.rescaled else fused.oft_bypass
        return partial(fn, linear, lyco, self.backend)

    def boft(self, linear, lyco, m):
        fn = fused.boft_bypass_rescaled if lyco.rescaled else fused.boft_bypass
        return partial(fn, linear, lyco, m, self.backend)

    def lora_unit(self, lyco):
        return partial(fused.lora_unit, lyco, self.backend)

    def loha_unit(self, lyco):
        if ops.hada_rank(lyco) > MAX_HADA_RANK:
            return None
        return partial(fused.loha_unit, lyco, self.backend)

    def lokr_unit(self, lyco):
        getters = fused.kron_getters(lyco, cast=False)
        return partial(fused.lokr_unit, lyco, getters, self.backend)

    def oft_unit(self, lyco):
        if lyco.block_size > MAX_BLOCK:
            return None
        fn = fused.oft_unit_rescaled if lyco.rescaled else fused.oft_unit
        return partial(fn, lyco, self.backend)

    def boft_unit(self, lyco, m):
        fn = fused.boft_unit_rescaled if lyco.rescaled else fused.boft_unit
        return partial(fn, lyco, m, self.backend)

    def decompose(self, lyco, on_out):
        return partial(fused.decompose, lyco, on_out, self.backend)


def _input_slot(op, width: int):
    return partial(flat, TierSlot(op, token_bucket, width=width))


def _weight_slot(op, shape: tuple):
    return TierSlot(op, static_bucket, shape=tuple(shape))


def _in(lyco) -> int:
    return int(lyco.shape[1])


def _out(lyco) -> int:
    return int(lyco.shape[0])


class Raced:
    """Every factory binds a TierSlot; the scope tests are LyCORIS's, inside the op."""

    def __init__(self, backend: str):
        self.backend = backend

    def lora(self, lyco, gamma):
        return _input_slot(ops.lora_bypass(lyco, gamma), _in(lyco))

    def loha(self, lyco, gamma):
        return _input_slot(ops.loha_bypass(lyco, gamma), _in(lyco))

    def lokr(self, lyco, scale):
        return _input_slot(ops.lokr_bypass(lyco, scale), _in(lyco))

    def oft(self, linear, lyco):
        slot = _input_slot(ops.oft_bypass(lyco), _out(lyco))
        return partial(fused.post_add, linear, slot)

    def boft(self, linear, lyco, m):
        slot = _input_slot(ops.boft_bypass(lyco, m), _out(lyco))
        return partial(fused.post_add, linear, slot)

    def lora_unit(self, lyco):
        return _weight_slot(ops.lora_merge(lyco), lyco.shape)

    def loha_unit(self, lyco):
        return _weight_slot(ops.loha_merge(lyco), lyco.shape)

    def lokr_unit(self, lyco):
        return _weight_slot(ops.lokr_merge(lyco), lyco.shape)

    def oft_unit(self, lyco):
        return _weight_slot(ops.oft_merge(lyco), lyco.shape)

    def boft_unit(self, lyco, m):
        return _weight_slot(ops.boft_merge(lyco, m), lyco.shape)

    def decompose(self, lyco, on_out):
        return _weight_slot(ops.decompose(lyco, on_out), lyco.shape)


def kit(backend: str, race: bool):
    """The kit for a fused `backend`: raced when the spec left the backend on auto."""
    return Raced(backend) if race else Pinned(backend)
