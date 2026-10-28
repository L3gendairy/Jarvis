"""Hardware fallback and Ollama startup tests. No GPU, microphone, speakers, network or LLM needed.

Run:  python test_hardware.py
"""
import logging
import shutil
from types import SimpleNamespace

import llm
from common import load_config
from hardware import Hardware, apply


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise SystemExit(1)


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(f"{record.levelname}: {record.getMessage()}")


def run(hw: Hardware, **stt) -> tuple[dict, list[str]]:
    cfg = load_config()
    cfg["stt"].update(stt)
    cap = Capture()
    logging.getLogger("jarvis.hardware").addHandler(cap)
    try:
        apply(cfg, hw)
    finally:
        logging.getLogger("jarvis.hardware").removeHandler(cap)
    return cfg["stt"], cap.lines


def main() -> None:
    logging.getLogger("jarvis.hardware").setLevel(logging.INFO)

    print("\n[1] No GPU -> CPU, int8, small")
    s, log = run(Hardware(note="no NVIDIA driver found"))
    check((s["device"], s["compute_type"], s["model"]) == ("cpu", "int8", "small"), f"stt -> {s['device']}/{s['compute_type']}/{s['model']}")
    check(any("Falling back to CPU" in l for l in log) and any("7B" in l for l in log), "fallback and 7B suggestion are logged")

    print("\n[2] Driver present but CUDA unusable -> same fallback")
    s, _ = run(Hardware(gpu_name="GeForce", vram_mib=12227, cuda_devices=0, note="driver too old"))
    check(s["device"] == "cpu" and s["model"] == "small", "cpu + small")

    print("\n[3] Already-small model is kept on CPU")
    s, _ = run(Hardware(), model="base")
    check(s["model"] == "base" and s["device"] == "cpu", "base stays base")

    print("\n[4] 12 GB GPU: nothing changes, no warning")
    s, log = run(Hardware(gpu_name="RTX 5070", vram_mib=12227, cuda_devices=1))
    check((s["device"], s["model"]) == ("cuda", "medium"), "still cuda/medium")
    check(not any("WARNING" in l for l in log), "no VRAM warning for a 12 GB card")

    print("\n[5] 8 GB GPU: CUDA kept, 7B suggested")
    s, log = run(Hardware(gpu_name="RTX 3060 Ti", vram_mib=8192, cuda_devices=1))
    check(s["device"] == "cuda", "still cuda")
    check(any("7B" in l for l in log), "7B suggestion logged")

    print("\n[6] Ollama not installed -> clear LLMError with the download link")
    real_which, real_find = shutil.which, llm._find_ollama
    try:
        shutil.which = lambda name, *a, **k: None
        llm._find_ollama = lambda: None
        j = llm.LLM(load_config())
        j._ping = lambda: False
        try:
            j.ensure_server()
            check(False, "should have raised")
        except llm.LLMError as e:
            check("not installed" in str(e) and "https://ollama.com/download" in str(e), str(e))
    finally:
        shutil.which, llm._find_ollama = real_which, real_find

    print("\n[7] Fresh PC: missing models are pulled (fake Ollama client, per-layer progress)")
    cfg = load_config()
    j = llm.LLM(cfg)
    pulled = []

    def pull(name, stream=True):
        pulled.append(name)
        for digest in ("sha256:a", "sha256:b"):
            for done in (0, 50, 100):
                yield SimpleNamespace(status="pulling", digest=digest, total=100, completed=done)

    j.client = SimpleNamespace(list=lambda: SimpleNamespace(models=[]), pull=pull)
    j.ensure_models()
    check(pulled == [cfg["llm"]["model"], cfg["llm"]["coder_model"]], f"pulled {pulled}")
    cfg["llm"]["auto_pull"] = False
    j2 = llm.LLM(cfg)
    j2.client = j.client
    try:
        j2.ensure_models()
        check(False, "should have raised")
    except llm.LLMError as e:
        check("ollama pull" in str(e), "auto_pull: false gives the manual command")

    print("\nAll hardware tests passed.")


if __name__ == "__main__":
    main()
