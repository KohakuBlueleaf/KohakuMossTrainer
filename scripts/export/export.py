"""Export a training state's adapters: ComfyUI / LyCORIS file, or a merged transformer.

kogine run scripts/export/export.py --set WEIGHTS=<model dir> --set ADAPTER=<run>/trainable.safetensors --set OUT=lora.safetensors
kogine run scripts/export/export.py --set FORMAT=merged --set OUT=merged.safetensors ...
"""

from kohakumoss.adapters.export.cli import RENAMED, export_main
from kohakumoss_common.knobs import check_script

WEIGHTS = ""  # base transformer of the model (HF dir, single file, or a ComfyUI models folder; any fold: the state's recorded fold is applied)
TRANSFORMER = ""  # ComfyUI folder: the DiT file name ("" = moss_image_<VARIANT>_int8_convrot, else _bf16)
VARIANT = "base"  # "base" | "sft": the ComfyUI DiT file
ADAPTER = ""  # a training state: a run checkpoint's trainable.safetensors or a LycoAdapterSet.save_state file
OUT = ""  # output file (or directory for FORMAT="diffusers")
FORMAT = "comfy"  # "comfy" | "hf" (= "lycoris") | "lycoris" | "merged" | "diffusers" (a full-finetune checkpoint: "comfy" = merged, "hf" = diffusers)
STYLE = "lora_unet"  # comfy/lycoris keys: "lora_unet" | "diffusion_model"
SAVE_DTYPE = "bf16"  # dtype of the written tensors: "bf16" | "fp16" | "fp32"
MULTIPLIER = 1.0  # strength baked into merged outputs
FUSED_QKV = "auto"  # comfy / merged: joint q/k/v as one proj.qkv: "auto" (as WEIGHTS stores them; the Comfy-Org files are fused) | True | False
DEVICE = "cpu"


def main():
    check_script(globals(), RENAMED)
    return export_main(globals())


if __name__ == "__main__":
    main()
