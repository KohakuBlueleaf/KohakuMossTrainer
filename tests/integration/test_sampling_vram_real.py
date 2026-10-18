"""Sampling VRAM of the low-memory layers on the real pipeline: one 1 MP image (4 steps,
CFG 4) with the DiT under `minimal` and under `low`, the text encoder streamed and the
VAE offloaded. Records the generate's torch allocated / reserved peaks and nvidia-smi
device peak (and the smi peak of the pipeline build) next to the card size each layer
targets (8 GiB / 12 GiB); the generate must finish with finite latents."""

import subprocess
import threading

import pytest
import torch

from kohakumoss.inference import MossPipeline
from kohakumoss_kernels.hardware.memory import close_window, open_window

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
GIB = 2**30
TARGET_GIB = {"minimal": 8, "low": 12}


class SmiPeak:
    """nvidia-smi memory.used of this process's device every 100 ms; `peak` in MiB."""

    def __init__(self):
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        ident = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
        cmd = ["nvidia-smi", f"--id={ident}", "--query-gpu=memory.used"]
        cmd += ["--format=csv,noheader,nounits", "-lms", "100"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        self.peak = 0
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            if line.strip().isdigit():
                self.peak = max(self.peak, int(line))

    def stop(self) -> int:
        self.proc.terminate()
        self.thread.join(timeout=5)
        return self.peak


@pytest.mark.parametrize("profile", list(TARGET_GIB))
def test_sampling_vram_of_the_low_layers(
    profile, weights_root, captions, record_property
):
    smi = SmiPeak()
    pipe = MossPipeline.from_pretrained(
        weights_root / "base",
        dtype=torch.bfloat16,
        device="cuda",
        vae_plan={"memory": "offload"},
        text_plan={"memory": "stream"},
        memory=profile,
        tune="off",
    )
    torch.cuda.synchronize()
    win = open_window("cuda")
    build_peak, smi.peak = smi.peak / 1024, 0
    res = pipe.generate(
        captions[:1],
        sizes=(1024, 1024),
        steps=4,
        cfg=4.0,
        seeds=[0],
        batch=1,
        output="latent",
        progress=False,
    )
    torch.cuda.synchronize()
    alloc, reserved = (b / GIB for b in close_window(win))
    device = smi.stop() / 1024
    del pipe
    torch.cuda.empty_cache()
    target = TARGET_GIB[profile]
    for key, value in (
        ("peak_alloc_gib", alloc),
        ("peak_reserved_gib", reserved),
        ("smi_peak_gib", device),
        ("smi_build_peak_gib", build_peak),
        ("target_card_gib", target),
        ("fits_target", device <= target),
    ):
        record_property(f"{profile}_{key}", value)
    print(
        f"{profile}: alloc {alloc:.2f} reserved {reserved:.2f} smi {device:.2f} GiB "
        f"(target {target} GiB)"
    )
    assert all(torch.isfinite(z).all() for z in res.latents)
    assert 0 < alloc <= reserved and device > 0, (alloc, reserved, device)
