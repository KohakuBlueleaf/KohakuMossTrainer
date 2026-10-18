"""Bench the fused linear + epilogue ops at MOSS shapes against F.linear + the eager
epilogue and torch.compile of that composition.

    python scripts/bench/kernel/fused_linear.py --backend triton
"""

import torch
import torch.nn.functional as F

from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels.backends.eager import ffn, norm, rope
from kohakumoss_kernels.bench.blockops import contenders, dtypes, measure, parse, report


def eager_situ(y, w, b):
    return ffn.situ_glu(F.linear(y, w, b))


def eager_res(a, w, b, x, gate, seg, r):
    return norm.gated_residual(x, F.linear(a, w, b), gate, seg, r)


def eager_qk(y, wq, wk, qw, kw, pos, bands, eps, heads):
    q = F.linear(y, wq).unflatten(-1, (heads, -1))
    k = F.linear(y, wk).unflatten(-1, (heads, -1))
    return rope.qk_norm_rope(q, k, qw, kw, pos, bands, eps)


def main():
    args = parse()
    device = torch.device(args.device)
    rows = []
    hw = [[64, 64]] * (1 if args.quick else 4)
    text_lens = [300, 120, 77, 500][: len(hw)]
    seq = build_seqinfo(torch.tensor(hw), torch.tensor(text_lens))
    n = seq.n_joint
    seg = seq.seg_joint.to(device)
    bands = (1 / (10000.0 ** (torch.arange(16) / 16))).to(device)
    bias = None
    for dtype in dtypes(args):
        y = torch.randn(n, 3072, device=device).to(dtype)
        w_ug = (torch.randn(16384, 3072, device=device) / 55).to(dtype)
        fns = contenders(
            "linear_situ", args.backend, eager_situ, device, dtype, not args.no_compile
        )
        rows.append(
            measure(
                f"situ N={n}",
                fns,
                eager_situ,
                (y, w_ug, bias),
                (True, False, False),
                dtype,
                args.iters,
                device,
            )
        )
        del w_ug
        a = torch.randn(n, 8192, device=device).to(dtype)
        w_dn = (torch.randn(3072, 8192, device=device) / 90).to(dtype)
        x = torch.randn(n, 3072, device=device).to(dtype)
        gate = torch.randn(len(hw), 3072, device=device)
        fns = contenders(
            "linear_gated_residual",
            args.backend,
            eager_res,
            device,
            dtype,
            not args.no_compile,
        )
        call = (a, w_dn, None, x, gate, seg, 1.0)
        rows.append(
            measure(
                f"ffn_down N={n}",
                fns,
                eager_res,
                call,
                (True, False, False, True, False, False, False),
                dtype,
                args.iters,
                device,
            )
        )
        w_o = (torch.randn(3072, 3072, device=device) / 55).to(dtype)
        call = (y, w_o, None, x, gate, seg, 1.0)
        rows.append(
            measure(
                f"o_proj N={n}",
                fns,
                eager_res,
                call,
                (True, False, False, True, False, False, False),
                dtype,
                args.iters,
                device,
            )
        )
        wq = (torch.randn(3072, 3072, device=device) / 55).to(dtype)
        wk = (torch.randn(3072, 3072, device=device) / 55).to(dtype)
        ones = torch.ones(128, device=device)
        fns = contenders(
            "linear_qk_rope", args.backend, eager_qk, device, dtype, not args.no_compile
        )
        call = (y, wq, wk, ones, ones, seq.pos_joint.to(device), bands, 1e-6, 24)
        rows.append(
            measure(
                f"qk N={n}",
                fns,
                eager_qk,
                call,
                (True, False, False, False, False, False, False, False, False),
                dtype,
                args.iters,
                device,
            )
        )
    report("fused_linear", rows, args.out, device)


if __name__ == "__main__":
    main()
