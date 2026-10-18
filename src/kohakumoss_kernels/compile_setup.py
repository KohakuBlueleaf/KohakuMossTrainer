"""Process-wide compile settings, applied before the first inductor compile: this
package's compile-cache key tag (`cache_tag`) and a compile worker pool that is never
quiesced. Every torch.compile site of the package calls `apply()` first. See
docs/internals/kernel-dev.md#compile-backend."""

import os
import sys

from kohakumoss_kernels import cache_tag

QUIESCE_ENV = "TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL"
INDUCTOR_CONFIG = "torch._inductor.config"
POOL_KEY = "quiesce_async_compile_pool"


def pool_setting() -> str:
    """The compile worker pool setting in force, for the run line."""
    value = os.environ.get(QUIESCE_ENV)
    return "no quiesce" if value == "0" else f"quiesce ({QUIESCE_ENV}={value})"


def apply() -> str:
    """Set the cache key tag and a never-quiesced compile pool for the whole process,
    every thread, and its child processes: TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0
    (an environment value already set wins), the value inductor's config reads from
    it put in place when that config is already loaded. Idempotent; returns the pool
    setting in force."""
    cache_tag.apply()
    if QUIESCE_ENV not in os.environ:
        os.environ[QUIESCE_ENV] = "0"
        loaded = sys.modules.get(INDUCTOR_CONFIG)
        if loaded is not None:
            loaded._config[POOL_KEY].env_value_force = False
    return pool_setting()
