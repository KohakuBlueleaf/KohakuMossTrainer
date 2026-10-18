"""Knob names the train and sample scripts share for replacement hints: the encoder
dict knobs' fields (an unknown `TEXT_<FIELD>` / `VAE_<FIELD>` is that field) and the
sampling knobs the two scripts name differently."""

from dataclasses import fields

from kohakumoss.offload.encoders.vae import VaePlan
from kohakumoss.types.plans import TextEncoderPlan

ENCODER_FIELDS = {
    "TEXT_": ("TEXT_ENCODE", tuple(f.name for f in fields(TextEncoderPlan))),
    "VAE_": ("VAE_ENCODE", tuple(f.name for f in fields(VaePlan))),
}
# train SAMPLE_<NAME> = sample <NAME>; train SAMPLE_SIZE = sample SIZES
SAMPLE_NAMES = ("PROMPTS", "SEEDS", "NEGATIVE", "SKIP_LAYERS", "CFG", "SHIFT")
RENAMED_SAMPLE = {
    "DTYPE": "renamed to PRECISION (the compute dtype)",
    "MEMORY": "renamed to MEMORY_PROFILE",
    "TEXT_PLAN": "renamed to TEXT_ENCODE",
    "VAE_PLAN": "renamed to VAE_ENCODE",
    "RESIDENCY": "removed: RESIDENCY='te_once' is TEXT_ENCODE['free_after_use'] = True",
    "SIZE": "renamed to SIZES (one size per prompt, the last one repeats)",
}
