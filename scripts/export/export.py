"""Export a training state's adapters: ComfyUI / LyCORIS file, or a merged transformer.

kogine run scripts/export/export.py --set WEIGHTS=<model dir> --set ADAPTER=<run>/trainable.safetensors --set OUT=lora.safetensors
kogine run scripts/export/export.py --set FORMAT=merged --set OUT=merged.safetensors ...
"""

from kohakumoss.adapters.export.cli import RENAMED, export_main
from kohakumoss_common.knobs import check_script

WEIGHTS = ""  # base transformer of the model (HF dir or single file; any fold: the state's recorded fold is applied)
ADAPTER = ""  # a training state: a run checkpoint's trainable.safetensors or a LycoAdapterSet.save_state file
OUT = ""  # output file (or directory for FORMAT="diffusers")
FORMAT = "comfy"  # "comfy" | "lycoris" | "merged" | "diffusers" (a full-finetune checkpoint: merged / diffusers)
STYLE = "lora_unet"  # comfy/lycoris keys: "lora_unet" | "diffusion_model"
SAVE_DTYPE = "bf16"  # dtype of the written tensors: "bf16" | "fp16" | "fp32"
MULTIPLIER = 1.0  # strength baked into merged outputs
FUSED_QKV = False  # comfy / merged: joint q/k/v as one proj.qkv (the Comfy-Org repack)
DEVICE = "cpu"


def main():
    check_script(globals(), RENAMED)
    return export_main(globals())


if __name__ == "__main__":
    main()
