"""Mid-run tuning across ranks: the lowest rank's measurement of each problem wins, every
rank settles and stores it, and a rank whose own choice differed drops its bindings."""

import pytest
import torch
import torch.distributed as dist

from kohakumoss_kernels.tuning import tuner, window
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
CARD = "card-a"
GATHERED = [
    [(CARD, "k", "p1", {"t": 0}), (CARD, "k", "p2", {"t": 0})],
    [(CARD, "k", "p1", {"t": 1}), (CARD, "k", "p3", {"t": 1}), ("card-b", "k", "p1", {"t": 1})],
]  # fmt: skip


def check_merge(agreed) -> None:
    got = agreed(GATHERED)
    assert got[(CARD, "k", "p1")] == {"t": 0}, "rank 0 measured p1 first"
    assert got[(CARD, "k", "p3")] == {"t": 1}, "only rank 1 measured p3"
    assert got[("card-b", "k", "p1")] == {"t": 1}
    assert len(got) == 4


def test_lowest_rank_wins():
    check_merge(tuner.agreed)


def test_negative_highest_rank_wins():
    must_fail(lambda: check_merge(lambda g: tuner.agreed(g[::-1])), "highest rank")


@pytest.fixture
def rank_state(tmp_path, monkeypatch):
    """This process has device "cpu" on CARD, chose {"t": 1} for p1 itself and holds a
    binding built from it."""
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuner, "_TABLES", {})
    monkeypatch.setattr(tuner, "_CARDS", {"cpu": CARD})
    monkeypatch.setattr(tuner, "_CHOSEN", {("cpu", "k", "p1"): {"t": 1}})
    bound = {"p1": "built from {'t': 1}"}
    tuner.on_reset(bound.clear)
    yield bound
    tuner._RESETS.remove(bound.clear)


def check_adopted(bound, adopt) -> None:
    changed = adopt(tuner.agreed(GATHERED))
    assert tuner._CHOSEN[("cpu", "k", "p1")] == {"t": 0}
    assert tuner._CHOSEN[("cpu", "k", "p3")] == {"t": 1}
    stored = tuner._card_table(CARD, "k")
    assert stored.get("p1") == {"t": 0}, "the winner was not stored"
    assert not bound, "a binding built from the replaced choice survived"
    assert changed == 3, changed


def test_adopt_rebinds(rank_state):
    check_adopted(rank_state, tuner.adopt)


def test_negative_adopt_keeps_bindings(rank_state, monkeypatch):
    monkeypatch.setattr(tuner, "_rebind", lambda: None)
    must_fail(lambda: check_adopted(rank_state, tuner.adopt), "bindings kept")


def test_agree_alone_is_a_no_op(tmp_path, monkeypatch):
    """One rank: nothing is exchanged; the pending measurements are dropped."""
    monkeypatch.setattr(tuner, "_NEW", [(CARD, "k", "p", {"t": 0})])
    dist.init_process_group(
        "gloo",
        init_method=f"file://{(tmp_path / 'store').as_posix()}",
        rank=0,
        world_size=1,
    )
    try:
        assert tuner.agree() == 0
    finally:
        dist.destroy_process_group()
    assert tuner._NEW == []
    tuner._NEW.append((CARD, "k", "p", {"t": 0}))
    assert tuner.agree() == 0 and tuner._NEW == [], "no process group"


def test_measured_rows_are_queued(tmp_path, monkeypatch):
    """A measured winner joins the rows the next `agree` exchanges."""
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuner, "_TABLES", {})
    monkeypatch.setattr(tuner, "_CHOSEN", {})
    monkeypatch.setattr(tuner, "_NEW", [])
    monkeypatch.setattr(tuner, "TIMINGS", {"events": lambda fn, dev: fn()})
    tuner.set_mode("table")
    cpu = torch.device("cpu")
    got = tuner.select(
        "k", cpu, "p", lambda: [{"ms": 2}, {"ms": 1}], lambda c: lambda: c["ms"]
    )
    card = tuner._CARDS[str(cpu)]
    assert tuner._NEW == [(card, "k", "p", got)], tuner._NEW


class _TwoRanks:
    """A torch.distributed stand-in for two gloo ranks: the other rank measured
    `others` problems; counts the object gathers."""

    def __init__(self, others: int):
        self.others, self.gathers = others, 0

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_world_size(self, group=None):
        return 2

    def get_backend(self, group=None):
        return "gloo"

    def all_reduce(self, t, group=None):
        t += self.others

    def all_gather_object(self, out, obj, group=None):
        self.gathers += 1
        out[0], out[1] = obj, []


def check_in_step_miss_deferred(monkeypatch, grouped: bool, kernel: str) -> None:
    """After go-live a key first met in a step is bound untimed and named in DEFERRED
    under two ranks; one process times it as before; the between-step retune
    (`timing_here`) times it under two ranks too."""
    timed = []

    def launch(config):
        return lambda: 1.0

    def measure(fn, device):
        timed.append(1)
        return fn()

    monkeypatch.setattr(tuner, "TIMINGS", {"events": measure})
    monkeypatch.setattr(tuner, "dist", _TwoRanks(0) if grouped else dist)
    monkeypatch.setattr(tuner, "_STATE", {**tuner._STATE, "mode": "table"})
    window.reopen()
    monkeypatch.setattr(tuner, "DEFERRED", set())
    monkeypatch.setattr(tuner, "_CHOSEN", {})
    tuner.decisions.go_live()
    cpu = torch.device("cpu")
    got = tuner.select(kernel, cpu, "p", lambda: [{"a": 1}, {"a": 2}], launch)
    assert got == {"a": 1}
    assert bool(timed) != grouped, ("timed in a step", timed)
    assert (("cpu", kernel, "p") in tuner.DEFERRED) == grouped
    with window.timing_here():
        tuner.select(kernel, cpu, "q", lambda: [{"a": 1}, {"a": 2}], launch)
    assert timed, "the between-step retune did not time"


def test_in_step_misses_wait_for_the_retune(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuner, "_TABLES", {})
    try:
        check_in_step_miss_deferred(monkeypatch, True, "k_two")
        check_in_step_miss_deferred(monkeypatch, False, "k_one")
        monkeypatch.setattr(tuner.decisions, "_LIVE_HOOKS", [lambda: None])
        must_fail(
            lambda: check_in_step_miss_deferred(monkeypatch, True, "k_broken"),
            "timed in a step",
        )
    finally:
        tuner.decisions.clear()


def check_deferred_skips_a_broken_config(monkeypatch, kernel: str) -> None:
    """Deferred under two ranks: a first config that cannot build is skipped, untimed,
    and the next one binds; every config failing raises ConfigsFailed."""
    from_launch = []

    def launch(config):
        def run():
            from_launch.append(config)
            if config.get("broken"):
                raise RuntimeError("triton: out of resources (test)")
            return 1.0

        return run

    monkeypatch.setattr(tuner, "dist", _TwoRanks(0))
    monkeypatch.setattr(tuner, "_STATE", {**tuner._STATE, "mode": "table"})
    window.reopen()
    monkeypatch.setattr(tuner, "DEFERRED", set())
    monkeypatch.setattr(tuner, "_CHOSEN", {})
    monkeypatch.setattr(tuner, "_FAILED", {})
    tuner.decisions.go_live()
    cpu = torch.device("cpu")
    ranked = [{"broken": 1}, {"a": 2}]
    assert tuner.select(kernel, cpu, "p", lambda: ranked, launch) == {"a": 2}
    assert ("cpu", kernel, "p") in tuner.DEFERRED
    with pytest.raises(tuner.ConfigsFailed):
        tuner.select(kernel, cpu, "q", lambda: [{"broken": 1}, {"broken": 2}], launch)


def _first_unchecked(ident, device, key, ranked, launch, errors):
    """The broken variant: the first candidate binds without running."""
    tuner._CHOSEN[ident] = ranked[0]
    tuner.DEFERRED.add(ident)
    return ranked[0]


def test_deferred_key_skips_a_broken_config(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuner, "_TABLES", {})
    try:
        check_deferred_skips_a_broken_config(monkeypatch, "k_def")
        monkeypatch.setattr(tuner, "_defer", _first_unchecked)
        must_fail(
            lambda: check_deferred_skips_a_broken_config(monkeypatch, "k_def2"),
            "the broken first config bound",
        )
    finally:
        tuner.decisions.clear()


def check_gathers_only_when_measured(monkeypatch) -> None:
    """No rank measured: no object gather; this rank or the other measured: one."""
    for others, mine, want in (
        (0, [], 0),
        (0, [(CARD, "k", "p", {"t": 0})], 1),
        (1, [], 1),
    ):
        fake = _TwoRanks(others)
        monkeypatch.setattr(tuner, "dist", fake)
        monkeypatch.setattr(tuner, "_NEW", list(mine))
        tuner.agree()
        assert fake.gathers == want, (others, mine, fake.gathers)


def test_agree_gathers_only_after_a_measurement(monkeypatch):
    check_gathers_only_when_measured(monkeypatch)
    monkeypatch.setattr(tuner, "measured_anywhere", lambda count, group=None: True)
    must_fail(
        lambda: check_gathers_only_when_measured(monkeypatch), "a gather every step"
    )
