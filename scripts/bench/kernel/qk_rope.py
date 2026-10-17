"""Bench qk_norm_rope at MOSS shapes (24 heads x 128, packed joint sequences).

python scripts/bench/kernel/qk_rope.py --backend triton
"""

import torch

from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels.backends.eager import rope as eager
from kohakumoss_kernels.bench.blockops import contenders, dtypes, measure, parse, report


def main():
    args = parse()
    device = torch.device(args.device)
    rows = []
    cases = [([[64, 64]], [300]), ([[64, 64]] * 4, [300, 120, 77, 500])]
    if args.quick:
        cases = cases[:1]
    bands = (1 / (10000.0 ** (torch.arange(16) / 16))).to(device)
    for dtype in dtypes(args):
        for hw, tl_ in cases:
            seq = build_seqinfo(torch.tensor(hw), torch.tensor(tl_))
            n = seq.n_joint
            q = torch.randn(n, 24, 128, device=device).to(dtype)
            k = torch.randn(n, 24, 128, device=device).to(dtype)
            qw = torch.ones(128, device=device)
            kw = torch.ones(128, device=device)
            fns = contenders(
                "qk_norm_rope",
                args.backend,
                eager.qk_norm_rope,
                device,
                dtype,
                not args.no_compile,
            )
            call = (q, k, qw, kw, seq.pos_joint.to(device), bands, 1e-6)
            for label, w_grad in (("lora", False), ("full", True)):
                grads = (True, True, w_grad, w_grad, False, False, False)
                rows.append(
                    measure(
                        f"N={n} B={len(hw)} {label}",
                        fns,
                        eager.qk_norm_rope,
                        call,
                        grads,
                        dtype,
                        args.iters,
                        device,
                    )
                )
    report("qk_norm_rope", rows, args.out, device)


if __name__ == "__main__":
    main()
