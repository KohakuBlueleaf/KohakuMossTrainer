"""Bench rms_norm_mod / rms_norm at MOSS shapes (D = 3072, packed 1 or 4 samples).

python scripts/bench/kernel/norm_mod.py --backend triton
"""

import torch

from kohakumoss_kernels.backends.eager import norm as eager
from kohakumoss_kernels.bench.blockops import contenders, dtypes, measure, parse, report


def seg_of(lens, device):
    return torch.cat(
        [torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lens)]
    ).to(device)


def main():
    args = parse()
    device = torch.device(args.device)
    rows = []
    cases = [(4400,), (4100, 4100, 4100, 4100)] if not args.quick else [(4400,)]
    for dtype in dtypes(args):
        for lens in cases:
            n, d = sum(lens), 3072
            x = torch.randn(n, d, device=device).to(dtype)
            w = torch.ones(d, device=device)
            shift = 0.1 * torch.randn(len(lens), d, device=device)
            scale = 0.1 * torch.randn(len(lens), d, device=device)
            seg = seg_of(lens, device)
            fns = contenders(
                "rms_norm_mod",
                args.backend,
                eager.rms_norm_mod,
                device,
                dtype,
                not args.no_compile,
            )
            call = (x, w, shift, scale, seg, 1e-6)
            name = f"mod N={n} B={len(lens)}"
            rows.append(
                measure(
                    name + " dx",
                    fns,
                    eager.rms_norm_mod,
                    call,
                    (True, False, False, False, False, False),
                    dtype,
                    args.iters,
                    device,
                )
            )
            rows.append(
                measure(
                    name + " all",
                    fns,
                    eager.rms_norm_mod,
                    call,
                    (True, True, True, True, False, False),
                    dtype,
                    args.iters,
                    device,
                )
            )
            fns = contenders(
                "rms_norm",
                args.backend,
                eager.rms_norm,
                device,
                dtype,
                not args.no_compile,
            )
            rows.append(
                measure(
                    f"norm N={n}",
                    fns,
                    eager.rms_norm,
                    (x, w.to(dtype), 1e-6),
                    (True, True, False),
                    dtype,
                    args.iters,
                    device,
                )
            )
    report("rms_norm_mod", rows, args.out, device)


if __name__ == "__main__":
    main()
