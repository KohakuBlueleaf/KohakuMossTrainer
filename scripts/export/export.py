"""Export a training state's adapters: ComfyUI / LyCORIS file, or a merged transformer.

kogine run scripts/export/export.py --set WEIGHTS=<model dir> --set ADAPTER=<run>/trainable.safetensors --set OUT=lora.safetensors
kogine run scripts/export/export.py --set FORMAT=merged --set OUT=merged.safetensors ...
"""

from kohakumoss.adapters.cli import export_main

WEIGHTS = ""  # base transformer the adapter was trained on (HF dir or single file)
ADAPTER = ""  # raw training state written by the trainer / LycoAdapterSet.save_state
OUT = ""  # output file (or directory for FORMAT="diffusers")
FORMAT = "comfy"  # "comfy" | "lycoris" | "merged" | "diffusers"
STYLE = "lora_unet"  # comfy/lycoris keys: "lora_unet" | "diffusion_model"
DTYPE = "bf16"  # "bf16" | "fp16" | "fp32"
MULTIPLIER = 1.0  # strength baked into merged outputs
FUSED_QKV = False  # merged: write joint q/k/v as one proj.qkv
FP16_FIX = "none"  # the run's FP16_FIX ("weights" folds the trained weights)
FP16_FIX_KWARGS = {}  # the run's FP16_FIX_KWARGS (c, s_attn, s_ffn, eps)
DEVICE = "cpu"

if __name__ == "__main__":
    export_main(globals())
