"""Qwen3.5 text encoder: tokens, placement plans, the kernel and transformers forwards."""

from kohakumoss.encoders.text.encoder import ENCODER_ID, QwenTextEncoder
from kohakumoss.encoders.text.factory import text_encoder
from kohakumoss.encoders.text.plan import check_plan, plan_problems, text_plan
from kohakumoss.encoders.text.tokens import MAX_LENGTH, tokenizer_dir

__all__ = [
    "ENCODER_ID",
    "MAX_LENGTH",
    "QwenTextEncoder",
    "check_plan",
    "plan_problems",
    "text_encoder",
    "text_plan",
    "tokenizer_dir",
]
