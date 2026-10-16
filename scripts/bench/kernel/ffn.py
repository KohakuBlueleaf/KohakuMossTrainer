"""Bench situ_glu at MOSS shapes ([N, 2 x 8192]).

python scripts/bench/kernel/ffn.py --backend triton
"""

import torch
from _blockops import contenders, dtypes, measure, parse, report

from kohakumoss_kernels.backends.eager import ffn as eager


def main():
    args = parse()
    device = torch.device("cuda")
    rows = []
    for dtype in dtypes(args):
        for n in (4400,) if args.quick else (4400, 16400):
            x = (4 * torch.randn(n, 16384, device=device)).to(dtype)
            fns = contenders(
                "situ_glu",
                args.backend,
                eager.situ_glu,
                device,
                dtype,
                not args.no_compile,
            )
            rows.append(
                measure(
                    f"N={n}",
                    fns,
                    eager.situ_glu,
                    (x,),
                    (True,),
                    dtype,
                    args.iters,
                    device,
                )
            )
    report("situ_glu", rows, args.out, device)


if __name__ == "__main__":
    main()
