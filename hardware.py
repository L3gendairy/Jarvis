"""Startup hardware detection: NVIDIA GPU / CUDA availability and VRAM, with CPU fallbacks instead of crashes.

Standalone:  python hardware.py     prints what was detected and what would be changed
"""
import logging
import shutil
import subprocess
from dataclasses import dataclass

log = logging.getLogger("jarvis.hardware")

# 12 GB cards report ~12227 MiB (11.94 GiB), so "under 12 GB" must not be compared against 12288 MiB.
LOW_VRAM_MIB = 11.5 * 1024
CPU_WHISPER_MODEL = "small"
# Whisper models that are already small enough for CPU use; anything else is replaced by CPU_WHISPER_MODEL.
_CPU_OK = ("tiny", "base", "small")


@dataclass
class Hardware:
    gpu_name: str = ""
    vram_mib: int = 0           # total VRAM of the largest NVIDIA GPU, 0 if none
    cuda_devices: int = 0       # devices CTranslate2 (faster-whisper) can use
    note: str = ""              # why CUDA is unavailable, if it is

    @property
    def cuda_ok(self) -> bool:
        return self.cuda_devices > 0


def detect() -> Hardware:
    hw = Hardware()
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=15).stdout
            best = (0, "")
            for line in out.strip().splitlines():
                name, _, mem = line.rpartition(",")
                if mem.strip().isdigit() and int(mem) > best[0]:
                    best = (int(mem), name.strip())
            hw.vram_mib, hw.gpu_name = best
        except (OSError, subprocess.SubprocessError) as e:
            hw.note = f"nvidia-smi failed ({e})"
    else:
        hw.note = "no NVIDIA driver found (nvidia-smi is missing)"
    try:
        import ctranslate2
        hw.cuda_devices = int(ctranslate2.get_cuda_device_count())
        if not hw.cuda_devices and hw.gpu_name:
            hw.note = "the NVIDIA driver is present but CTranslate2 sees no CUDA device (driver too old?)"
        elif not hw.cuda_devices and not hw.note:
            hw.note = "no CUDA device found"
    except Exception as e:                       # ImportError, or a broken CUDA runtime
        hw.cuda_devices = 0
        hw.note = hw.note or f"CUDA check failed ({e})"
    return hw


def apply(cfg: dict, hw: Hardware | None = None) -> Hardware:
    """Adjust `cfg` in memory (config.yaml is not touched) so a PC without CUDA runs on the CPU, and log advice."""
    hw = hw or detect()
    stt = cfg.setdefault("stt", {})
    if hw.gpu_name:
        log.info("GPU: %s, %.1f GB VRAM, CUDA %s", hw.gpu_name, hw.vram_mib / 1024,
                 "available" if hw.cuda_ok else "NOT available")
    if not hw.cuda_ok:
        before = f"{stt.get('model', 'medium')} on {stt.get('device', 'cuda')}"
        stt["device"] = "cpu"
        stt["compute_type"] = "int8"
        if str(stt.get("model", "medium")).split(".")[0] not in _CPU_OK:
            stt["model"] = CPU_WHISPER_MODEL
        log.warning("No usable NVIDIA GPU/CUDA (%s). Falling back to CPU speech recognition: "
                    "stt.device=cpu, compute_type=int8, model=%s (was %s).", hw.note, stt["model"], before)
        log.warning("Ollama will also run on the CPU, which is slow with 14B models. "
                    "Consider the 7B models in config.yaml: llm.model: qwen2.5:7b, llm.coder_model: qwen2.5-coder:7b.")
    elif hw.vram_mib < LOW_VRAM_MIB:
        log.warning("Only %.1f GB of VRAM. The 14B models plus Whisper will not fit well; "
                    "suggestion: use the 7B models in config.yaml (llm.model: qwen2.5:7b, "
                    "llm.coder_model: qwen2.5-coder:7b).", hw.vram_mib / 1024)
    return hw


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from common import load_config
    c = load_config()
    h = detect()
    print(h)
    apply(c, h)
    print("stt ->", {k: c["stt"].get(k) for k in ("device", "compute_type", "model")})
