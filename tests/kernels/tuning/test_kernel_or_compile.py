"""Kernel-vs-compile binding: looked up in the call, measured only by retune_pending."""

import pytest
import torch
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
from torch.utils.checkpoint import checkpoint

from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.compile import choice
from kohakumoss_kernels.backends.compile.ops import compiled_op
from kohakumoss_kernels.backends.eager import ffn, norm, rope
from kohakumoss_kernels.tuning import tuner
from tests._util.blockops_prewarm import prewarm_block_ops
from tests._util.error_stats import assert_no_regression
from tests._util.negative import must_fail

TEST = "kernel_or_compile"


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    mode = tuner.get_mode()
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    choice.reset()
    yield
    tuner.set_mode(mode)
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    choice.reset()


def _slow_kernel(x):
    for _ in range(50):
        y = ffn.situ_glu(x)
    return y


@pytest.mark.gpu
def test_call_never_times_and_retune_binds_winner(device, fresh):
    tuner.set_mode("table")
    op = choice.kernel_or_compile("test", "situ_glu", _slow_kernel, choice.shape_key)
    x = torch.randn(2048, 8192, device=device)
    op(x)
    assert op.bound and next(iter(op.bound.values())) is op.impls["kernel"]
    assert op.pending, "first sight of a problem is queued, not timed"
    assert choice.retune_pending() >= 1
    assert op.choice == "compile"
    assert_no_regression(op(x), ffn.situ_glu(x), ffn.situ_glu(x.double()), "compile path", TEST)  # fmt: skip
    bound = op.bound[choice._call_key((x,))]
    assert bound is not op.impls["kernel"], "the kernel stayed bound"


@pytest.mark.gpu
@pytest.mark.parametrize("ctx", ["inference", "checkpoint"])
def test_retune_inside_untimed_context_stays_queued(device, fresh, ctx):
    """A retune reached under inference mode or inside a checkpointed forward times
    nothing and binds nothing; the next retune outside binds the measured winner."""
    tuner.set_mode("table")
    op = choice.kernel_or_compile("test", "situ_glu", _slow_kernel, choice.shape_key)
    x = torch.randn(1024, 4096, device=device, requires_grad=True)
    op(x)
    assert op.pending

    def inner(t):
        assert choice.retune_pending() == 0
        return t * 1

    if ctx == "inference":
        with torch.inference_mode():
            inner(torch.ones(1, device=device))
    else:
        checkpoint(inner, x, use_reentrant=False).sum().backward()
    assert op.pending and not tuner._CHOSEN
    assert choice.retune_pending() == 1
    assert op.choice == "compile" and not op.pending


@pytest.mark.gpu
def test_tune_off_never_measures(device, fresh):
    tuner.set_mode("off")
    op = choice.kernel_or_compile("test", "situ_glu", _slow_kernel, choice.shape_key)
    op(torch.randn(256, 512, device=device))
    assert not op.pending and choice.retune_pending() == 0


def _registered(op, device, dtype):
    try:
        return registry.resolve(op, device, dtype, backend="triton")
    except registry.NoImplementationError:
        return registry.resolve(op, device, dtype, backend="tilelang")


def _block_inputs(device, dtype):
    seq = build_seqinfo(torch.tensor([[4, 6], [3, 5]]), torch.tensor([7, 11]))
    n, d = seq.n_joint, 256
    x = torch.randn(n, d, device=device).to(dtype)
    mod = (0.1 * torch.randn(2, 6, d, device=device))[:, 3]
    bands = (1 / (10000.0 ** (torch.arange(16) / 16))).to(device)
    return x, mod, seq.seg_joint.to(device), seq.pos_joint.to(device), bands


def _block(ops, x, mod, seg, pos, bands):
    n = x.shape[0]
    y = ops["rms_norm_mod"](x, None, mod, mod, seg, 1e-6)
    q, k = ops["qk_norm_rope"](
        y.view(n, 2, 128), y.view(n, 2, 128).flip(-1).contiguous(),
        torch.ones(128, device=x.device), torch.ones(128, device=x.device), pos, bands, 1e-6,
    )  # fmt: skip
    a = ops["situ_glu"](torch.cat((q.reshape(n, -1), k.reshape(n, -1)), -1))
    return ops["gated_residual"](x, a, mod, seg, 1.0)


EAGER = {
    "rms_norm_mod": norm.rms_norm_mod,
    "qk_norm_rope": rope.qk_norm_rope,
    "situ_glu": ffn.situ_glu,
    "gated_residual": norm.gated_residual,
}


@pytest.mark.gpu
def test_checkpointed_block_on_fresh_cache(device, fresh):
    """Default kernels inside non-reentrant checkpointing, first step on a fresh cache."""
    dtype = torch.float16
    ops = {name: _registered(name, device, dtype) for name in EAGER}
    x, mod, seg, pos, bands = _block_inputs(device, dtype)
    xa = x.detach().clone().requires_grad_(True)
    out = checkpoint(_block, ops, xa, mod, seg, pos, bands, use_reentrant=False)
    out.float().square().sum().backward()
    xe = x.detach().clone().requires_grad_(True)
    _block(EAGER, xe, mod, seg, pos, bands).float().square().sum().backward()
    xr = x.detach().double().requires_grad_(True)
    _block(EAGER, xr, mod.double(), seg, pos, bands.double()).square().sum().backward()
    assert_no_regression(xa.grad, xe.grad, xr.grad, "checkpointed block, fresh cache", TEST)  # fmt: skip
    choice.retune_pending()
    xb = x.detach().clone().requires_grad_(True)
    checkpoint(
        _block, ops, xb, mod, seg, pos, bands, use_reentrant=False
    ).float().square().sum().backward()
    assert_no_regression(xb.grad, xe.grad, xr.grad, "checkpointed block, retuned", TEST)


SHAPES = (  # three distinct token counts (31, 57, 115), ending on the first again
    ([[4, 6]], [7]),
    ([[4, 6], [3, 5]], [7, 11]),
    ([[8, 8], [2, 2], [5, 3]], [3, 9, 18]),
    ([[4, 6]], [7]),
)


def _shape_inputs(hw, text, device, dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    seq = build_seqinfo(torch.tensor(hw), torch.tensor(text))
    x = torch.randn(seq.n_joint, 256, generator=gen).to(device, dtype)
    mod = (0.1 * torch.randn(len(hw), 256, generator=gen)).to(device)
    return x, mod, seq.seg_joint.to(device), seq.pos_joint.to(device)


def _accumulated_grads(ops, batch, bands):
    """Two checkpointed blocks per micro-batch; every forward first, then the backwards
    in reverse, so a recompute runs after later shapes went through the compiled ops."""
    xs = [b[0].detach().clone().requires_grad_(True) for b in batch]
    losses = []
    for x, (_, mod, seg, pos) in zip(xs, batch):
        h = x
        for _ in range(2):
            h = checkpoint(_block, ops, h, mod, seg, pos, bands, use_reentrant=False)
        losses.append(h.float().square().mean())
    for loss in reversed(losses):
        loss.backward()
    return [x.grad for x in xs]


def _graphs_per_pattern(op) -> int:
    """The most compiled graphs any one call pattern of a CompiledOp holds."""
    return max(
        (len(_debug_get_cache_entry_list(f.__code__)) for f in op.functions.values()),
        default=0,
    )


def _compile_path_steps(device, dynamic) -> dict[str, int]:
    """Compile path forced on every op; grads of each step vs the eager references.
    Returns the most compiled graphs one call pattern of each op holds."""
    dtype = torch.float16
    torch._dynamo.reset()
    ops = {name: _registered(name, device, dtype) for name in EAGER}
    registered = {name: op.impls["compile"] for name, op in ops.items()}
    bands = (1 / (10000.0 ** (torch.arange(16) / 16))).to(device)
    try:
        for name, op in ops.items():
            op.bind("compile")
            op.impls["compile"] = compiled_op(name, dynamic)
        for step in range(len(SHAPES)):
            batch = [
                _shape_inputs(
                    *SHAPES[(step + j) % len(SHAPES)], device, dtype, step + j
                )
                for j in range(2)
            ]
            got = _accumulated_grads(ops, batch, bands)
            eager = _accumulated_grads(EAGER, batch, bands)
            batch64 = [(x.double(), m.double(), s, p) for x, m, s, p in batch]
            ref = _accumulated_grads(EAGER, batch64, bands.double())
            for i, (g, e, r) in enumerate(zip(got, eager, ref)):
                assert_no_regression(g, e, r, f"compile path step {step} grad{i}", TEST)
        return {name: _graphs_per_pattern(compiled_op(name, dynamic)) for name in EAGER}
    finally:
        for name, op in ops.items():
            op.impls["compile"] = registered[name]
        torch._dynamo.reset()


def _check_graphs(graphs: dict[str, int]) -> None:
    """A token-count change never compiles a new graph: one graph per call pattern, two
    for the ops taking a per-sample tensor (one sample specialises its leading dim to
    1)."""
    per_sample = {"rms_norm_mod", "gated_residual"}
    for name, n in graphs.items():
        assert n == (2 if name in per_sample else 1), f"{name}: {n} graphs"


@pytest.mark.gpu
def test_compile_path_checkpointed_across_shapes(device, fresh):
    """Registered compile path (dynamic=True) inside block checkpointing, token count
    changing every step and across micro-batches whose recomputes run after later
    shapes: no checkpoint mismatch, eager grads, and no recompile on a new length."""
    _check_graphs(_compile_path_steps(device, True))


@pytest.mark.gpu
def test_compile_path_automatic_dynamic_is_caught(device, fresh):
    """`dynamic=None` (static first trace, dynamic recompile on the next length) is
    what lets a recompute meet a different graph than its forward; the graph check
    above must reject it."""
    graphs = _compile_path_steps(device, None)
    must_fail(lambda: _check_graphs(graphs), "automatic dynamic recompiles")


@pytest.mark.gpu
@pytest.mark.budget(20)
def test_prewarm_binds_every_op(device, fresh):
    tuner.set_mode("table")
    ops = {name: _registered(name, device, torch.float16) for name in EAGER}
    measured = prewarm_block_ops(
        ops, device, torch.float16, [[300, 200]], dim=256, heads=2, ffn_dim=512
    )
    assert measured == len(ops)
    for op in ops.values():
        assert op.bound and not op.pending


@pytest.fixture
def two_backends(fresh):
    """situ_glu on cpu from two fake kernel backends (first, second) in the cpu
    priority, each counting its calls."""
    tuner.set_mode("table")
    order = registry._STATE.priority["cpu"]
    calls = {"first": 0, "second": 0}

    def counted(name):
        def kernel(x):
            calls[name] += 1
            return ffn.situ_glu(x)

        return kernel

    for name in calls:
        op = choice.kernel_or_compile(
            f"fake_{name}", "situ_glu", counted(name), choice.shape_key
        )
        registry.register("situ_glu", f"fake_{name}", op)
    registry.set_priority("cpu", ("fake_first", "fake_second", "compile", "eager"))
    yield calls
    registry.set_priority("cpu", order)
    for name in calls:
        registry._STATE.impls["situ_glu"].pop(f"fake_{name}", None)


def check_most_prior_only(op, calls) -> None:
    """Only the most-prior backend's kernel and compile are measured: the second
    backend is never built, probed or timed."""
    x = torch.randn(64, 32)
    assert op.backend == "fake_first"
    op(x)
    op.decide(x)
    assert calls["second"] == 0, calls
    assert calls["first"] > 0


@pytest.mark.cpu
def test_choice_tunes_the_most_prior_backend_only(two_backends):
    check_most_prior_only(
        registry.resolve("situ_glu", "cpu", torch.float32), two_backends
    )


@pytest.mark.cpu
def test_negative_sibling_kernel_is_caught(two_backends, monkeypatch):
    """A choice that also times the next backend's kernel must fail the check."""
    second = registry._STATE.impls["situ_glu"]["fake_second"].fn.__wrapped__
    monkeypatch.setattr(choice, "CHOICES", ("kernel", "second", "compile"))
    op = registry.resolve("situ_glu", "cpu", torch.float32)
    op.fn.impls["second"] = second
    must_fail(lambda: check_most_prior_only(op, two_backends), "sibling kernel timed")


@pytest.mark.gpu
@pytest.mark.parametrize("ctx", [torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_no_grad_forward_on_fresh_cache(device, fresh, ctx, dtype):
    try:
        ops = {name: _registered(name, device, dtype) for name in EAGER}
    except registry.NoImplementationError as err:
        pytest.skip(str(err))
    x, mod, seg, pos, bands = _block_inputs(device, dtype)
    with ctx():
        out = _block(ops, x, mod, seg, pos, bands)
        eager = _block(EAGER, x, mod, seg, pos, bands)
        ref = _block(EAGER, x.double(), mod.double(), seg, pos, bands.double())
    assert_no_regression(out, eager, ref, f"no-grad block {dtype}", TEST)
