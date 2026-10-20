"""Knob checks of the training text encoder and VAE placement."""

from kohakumoss.encoders.text.encoder import stored_text_format
from kohakumoss.encoders.text.plan import STORED_ONLY, text_plan
from kohakumoss.encoders.text.plan import plan_problems as text_plan_problems
from kohakumoss.encoders.vae import vae_dtype_problems
from kohakumoss.offload.accel import default_type
from kohakumoss.offload.encoders.vae import plan_problems as vae_plan_problems
from kohakumoss.offload.encoders.vae import vae_plan
from kohakumoss.training.weights import component
from kohakumoss.weights import TEXT
from kohakumoss_kernels.hardware.arch import arch_of


def vae_problems(k: dict) -> list[str]:
    """VAE_ENCODE (a VaePlan dict) and VAE_DTYPE."""
    try:
        plan = vae_plan(k.get("VAE_ENCODE") or {})
    except (TypeError, ValueError) as err:
        return [f"VAE_ENCODE: {err}"]
    out = [f"VAE_ENCODE: {m}" for m in vae_plan_problems(plan)]
    kind = default_type()
    index = (k.get("DEVICES") or (0,))[0]
    device = "cpu" if kind == "cpu" else f"{kind}:{index}"
    return out + vae_dtype_problems(k.get("VAE_DTYPE", "auto"), device)


def text_problems(k: dict, arch=arch_of) -> list[str]:
    """TEXT_ENCODE (a TextEncoderPlan dict) and TEXT_DEVICE; TEXT_DEVICE='cpu' means
    memory 'cpu'; `arch` reads a device's ArchInfo."""
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
    problems = text_plan_problems(plan, target, arch=arch)
    problems += stored_text_problems(k, plan)
    return [f"TEXT_ENCODE / TEXT_DEVICE: {m}" for m in problems]


def stored_text_problems(k: dict, plan) -> list[str]:
    """A stored-only precision (`w4a8`) the run's text-encoder weights do not store."""
    if plan.precision not in STORED_ONLY or not k.get("WEIGHTS"):
        return []
    try:
        path = component(k, TEXT)
    except FileNotFoundError:
        return []
    stored, _ = stored_text_format(path, plan)
    if stored == plan.precision:
        return []
    return [
        f"precision {plan.precision!r} is a stored format and {path} stores "
        f"{stored!r}: use 'auto' or a precision KohakuMoss quantizes to"
    ]
