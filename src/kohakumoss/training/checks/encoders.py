"""Knob checks of the training text encoder and VAE placement."""

from kohakumoss.encoders.text.plan import plan_problems as text_plan_problems
from kohakumoss.encoders.text.plan import text_plan
from kohakumoss.encoders.vae import vae_dtype_problems
from kohakumoss.offload.accel import default_type
from kohakumoss.offload.encoders.vae import plan_problems as vae_plan_problems
from kohakumoss.offload.encoders.vae import vae_plan
from kohakumoss_kernels.hardware.arch import arch_of


def vae_problems(k: dict) -> list[str]:
    """VAE_ENCODE (a VaePlan dict); memory 'cpu' only with the offline latent cache."""
    try:
        plan = vae_plan(k.get("VAE_ENCODE") or {})
    except (TypeError, ValueError) as err:
        return [f"VAE_ENCODE: {err}"]
    problems = vae_plan_problems(plan, training=True, latent_cache=k["LATENT_CACHE"])
    out = [f"VAE_ENCODE: {m}" for m in problems]
    kind = default_type()
    index = (k.get("DEVICES") or (0,))[0]
    device = "cpu" if kind == "cpu" else f"{kind}:{index}"
    return out + vae_dtype_problems(k.get("VAE_DTYPE", "auto"), device)


def text_problems(k: dict, arch=arch_of) -> list[str]:
    """TEXT_ENCODE (a TextEncoderPlan dict) and TEXT_DEVICE; TEXT_DEVICE='cpu' means
    memory 'cpu', which training allows only with the offline text cache; `arch` reads
    a device's ArchInfo."""
    spec = dict(k.get("TEXT_ENCODE") or {})
    device = str(k.get("TEXT_DEVICE") or "")
    if device.startswith("cpu"):
        spec["memory"] = "cpu"
    try:
        plan = text_plan(spec)
    except ValueError as err:
        return [f"TEXT_ENCODE: {err}"]
    target = None
    if not device.startswith("cpu") and default_type() == "cuda":
        target = device or "cuda:0"
    problems = text_plan_problems(
        plan, target, training=True, text_cache=k["TEXT_CACHE"], arch=arch
    )
    return [f"TEXT_ENCODE / TEXT_DEVICE: {m}" for m in problems]
