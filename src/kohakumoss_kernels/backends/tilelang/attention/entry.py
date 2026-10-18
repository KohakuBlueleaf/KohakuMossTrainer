"""The TileLang `attention_varlen` registry entry (sm70+): the TileLang kernels chosen
per shape against torch's varlen attention (`backends/varlen_choice.py`)."""

from kohakumoss_kernels.backends.tilelang.attention import varlen
from kohakumoss_kernels.backends.varlen_choice import varlen_choice
from kohakumoss_kernels.registry import register

CHOICE = varlen_choice("tilelang", varlen.attention_varlen, varlen._key)
register("attention_varlen", "tilelang", CHOICE, varlen.SUPPORTS)
