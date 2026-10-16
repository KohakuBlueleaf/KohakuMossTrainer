"""Bench gated_residual ([N, 3072], per-sample gate) and token_mse ([N_img, 128]).

python scripts/bench/kernel/residual.py --backend triton
"""

import torch
from _blockops import contenders, dtypes, measure, parse, report

from kohakumoss_kernels.backends.eager import loss, norm


def seg_of(lens, device):
    return torch.cat(
        [torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lens)]
    ).to(device)


def main():
    args = parse()
    device = torch.device("cuda")
    rows, loss_rows = [], []
    cases = [(4400,), (4100, 4100, 4100, 4100)] if not args.quick else [(4400,)]
    for dtype in dtypes(args):
        for lens in cases:
            n, d = sum(lens), 3072
            seg = seg_of(lens, device)
            x = torch.randn(n, d, device=device).to(dtype)
            o = torch.randn(n, d, device=device).to(dtype)
            gate = torch.randn(len(lens), d, device=device)
            fns = contenders(
                "gated_residual",
                args.backend,
                norm.gated_residual,
                device,
                dtype,
                not args.no_compile,
            )
            call = (x, o, gate, seg, 1.0)
            rows.append(
                measure(
                    f"N={n} B={len(lens)} lora",
                    fns,
                    norm.gated_residual,
                    call,
                    (True, True, False, False, False),
                    dtype,
                    args.iters,
                    device,
                )
            )
            rows.append(
                measure(
                    f"N={n} B={len(lens)} gate",
                    fns,
                    norm.gated_residual,
                    call,
                    (True, True, True, False, False),
                    dtype,
                    args.iters,
                    device,
                )
            )
            img = [v * 4096 // 4400 for v in lens]
            p = torch.randn(sum(img), 128, device=device).to(dtype)
            t = torch.randn(sum(img), 128, device=device).to(dtype)
            w = torch.rand(len(lens), device=device)
            fns = contenders(
                "token_mse",
                args.backend,
                loss.token_mse,
                device,
                dtype,
                not args.no_compile,
            )
            loss_rows.append(
                measure(
                    f"N={sum(img)}",
                    fns,
                    loss.token_mse,
                    (p, t, seg_of(img, device), w, None),
                    (True, False, False, False, False),
                    dtype,
                    args.iters,
                    device,
                )
            )
    report("gated_residual", rows, args.out, device)
    report("token_mse", loss_rows, args.out, device)


if __name__ == "__main__":
    main()
