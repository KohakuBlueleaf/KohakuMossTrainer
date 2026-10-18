"""Model FLOP estimate per training step and nominal tensor-core peaks for MFU. See
docs/performance/bench-matrix.md (nominal peaks)."""

import torch

from kohakumoss.types.arch import MossArchConfig

PEAK_TFLOPS = (
    ("V100", 125.0),
    ("PG503", 125.0),
    ("2080 Ti", 53.8),
    ("Tesla T4", 65.0),
    ("Tesla T10", 68.8),
    ("3060", 25.6),
    ("3090", 71.0),
    ("4090", 165.2),
    ("5090", 209.5),
    ("RTX PRO 6000 Blackwell", 500.0),
    ("A100", 312.0),
    ("H100 PCIe", 756.0),
    ("H100", 989.0),
    ("NVIDIA L4", 121.0),
)
TRAIN_FACTOR = {"full": 3.0, "adapter": 2.0}


def forward_flops(
    cfg: MossArchConfig, img_tokens: int, joint_lens: list[int], text_tokens: int
) -> float:
    """Dense-layer + attention FLOPs of one forward over a packed batch."""
    d, f = cfg.dim, cfg.ffn_dim
    joint = img_tokens + text_tokens
    block_linear = 2 * joint * d * (4 * d + 3 * f)
    block_attn = 4 * d * sum(n * n for n in joint_lens)
    text_linear = 2 * text_tokens * d * (4 * d + 3 * f) * cfg.text_blocks
    embed = (
        2 * img_tokens * cfg.latent_channels * d * 2
        + 2 * text_tokens * cfg.text_dim * d
    )
    return float(cfg.depth * (block_linear + block_attn) + text_linear + embed)


def step_flops(fwd: float, mode: str, recompute: bool) -> float:
    """Training FLOPs: forward x factor (+1 forward when activations are recomputed)."""
    return fwd * (TRAIN_FACTOR[mode] + (1.0 if recompute else 0.0))


def peak_tflops(device: torch.device) -> float | None:
    """Nominal dense fp16/bf16 tensor-core peak (fp32 accumulate), None when unknown."""
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    for key, value in PEAK_TFLOPS:
        if key in name:
            return value
    return None
