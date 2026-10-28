"""Behaviour tests that need neither a microphone, speakers nor an LLM:
config defaults, input-device selection and /mic diagnostics, VRAM report, /status, inactivity timer, confirmation timer.

Run:  python test_jarvis.py
"""
import contextlib
import io
import logging
import re
import time
from types import SimpleNamespace

import numpy as np

import stt
from common import load_config, resolve, setup_logging
from main import HELP, Jarvis


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise SystemExit(1)


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def text(self, level=0):
        return [r.getMessage() for r in self.records if r.levelno >= level]


def main() -> None:
    setup_logging("INFO")
    cfg = load_config()

    print("\n[1] Config defaults")
    check(cfg["wake_word"]["inactivity_timeout_s"] == 300, "inactivity_timeout_s defaults to 300")
    check(not {"travis", "gervais"} & set(a.lower() for a in cfg["wake_word"]["aliases"]), "travis/gervais removed from aliases")
    check(cfg["audio"]["input_device"] == "", 'audio.input_device defaults to "" (system default)')
    text = open(resolve("config.yaml"), encoding="utf-8").read()
    check(re.search(r"^\s*#\s*model:\s*qwen2\.5:7b", text, re.M) and re.search(r"^\s*#\s*coder_model:\s*qwen2\.5-coder:7b", text, re.M),
          "7B alternatives are present as comments")
    check("/mic" in HELP and "/status" in HELP, "/help lists /mic and /status")

    print("\n[2] Input device selection (device list only; nothing is recorded)")
    devices = stt.list_input_devices()
    check(len(devices) > 0 and all({"index", "name", "hostapi", "channels"} <= d.keys() for d in devices), f"{len(devices)} input devices listed")
    idx, name = stt.resolve_input_device("")
    check(idx is None and name, f"'' -> system default ({name})")
    first = devices[0]
    word = first["name"].split()[0].lower() if first["name"].split() else first["name"]
    idx, name = stt.resolve_input_device(word.upper())
    check(word in name.lower(), f"partial, case-insensitive name '{word.upper()}' -> [{idx}] {name}")
    idx, name = stt.resolve_input_device(str(first["index"]))
    check(idx == first["index"], f"index '{first['index']}' -> {name}")
    for bad in ("definitely-not-a-microphone", "9999"):
        try:
            stt.resolve_input_device(bad)
            check(False, f"'{bad}' should raise")
        except stt.MicrophoneError as e:
            check("Run /mic" in str(e) or "Available" in str(e), f"unknown device '{bad}' -> clear error")
    solo = [d for d in devices if "solocast" in d["name"].lower()]
    if solo:
        idx, name = stt.resolve_input_device("SoloCast")
        check("solocast" in name.lower(), f"'SoloCast' -> [{idx}] {name} (your mic)")

    print("\n[3] /mic diagnostics")
    status, advice = stt.diagnose_level(0.0, 0.0)
    check(status == "silent" and all(k in advice for k in ("Hardware mute", "Windows privacy", "Wrong device")),
          "digital silence -> SILENT with hardware-mute / Windows-privacy / wrong-device fixes")
    check(stt.diagnose_level(0.004, 0.001)[0] == "quiet", "tiny signal -> QUIET")
    check(stt.diagnose_level(0.4, 0.05)[0] == "ok", "normal signal -> OK")
    fake_mic = SimpleNamespace(device_name="Fake Mic", measure=lambda s: (0.0, 0.0))
    res = stt.measure_level("", 0.0, live_mic=fake_mic)
    check(res["status"] == "silent" and res["device"] == "Fake Mic", "measure_level on a live (silent) mic reports silent")

    # a silent stream must NOT trigger any automatic warning (no diagnostics at startup)
    cap = Capture()
    logging.getLogger("jarvis.stt").addHandler(cap)
    m = stt.Microphone(cfg)
    import threading
    threading.Thread(target=m._process_loop, daemon=True).start()
    for _ in range(80):
        m._blocks.put(np.zeros(m.block, np.float32))
    time.sleep(0.8)
    m._stop.set()
    check(not cap.text(logging.WARNING), "silent audio produces no automatic warning at startup (only /mic diagnoses)")

    cfg2 = load_config()
    cfg2["settings_file"] = "data/tj_settings.json"
    cfg2["memory"]["db_path"] = "data/tj_memory.db"
    j = Jarvis(cfg2, text_only=True)
    j.tts._muted = True                                   # silent
    said = []
    orig_say = j.tts.say
    j.tts.say = lambda t: (said.append(t), orig_say(t))[0]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        j.cmd_mic()
    out = buf.getvalue()
    check("Input devices" in out and "IN USE" in out and "Text-only mode" in out,
          "/mic lists devices, marks the one in use, and skips measuring in --text mode")

    print("\n[4] VRAM report (Ollama /api/ps, faked)")
    def fake_ps(size, vram):
        return SimpleNamespace(models=[SimpleNamespace(model="qwen2.5:14b", size=size, size_vram=vram)])
    cap = Capture()
    logging.getLogger("jarvis.llm").addHandler(cap)
    j.llm.client = SimpleNamespace(ps=lambda: fake_ps(9_000_000_000, 6_300_000_000))
    j.llm.gpu_report()
    warn = " ".join(cap.text(logging.WARNING))
    check("only 70% on the GPU" in warn and "qwen2.5:7b" in warn and "qwen2.5-coder:7b" in warn,
          "70% on GPU -> warning that suggests the 7B models")
    cap.records.clear()
    j.llm.client = SimpleNamespace(ps=lambda: fake_ps(9_000_000_000, 9_000_000_000))
    j.llm.gpu_report()
    check(not cap.text(logging.WARNING) and any("GPU 100%" in m for m in cap.text()), "100% on GPU -> logged, no warning")

    print("\n[5] /status")
    s = j.status_text()
    check("ASLEEP" in s and "muted: yes" in s and str(j.code.root) in s, "shows asleep, mute state, workspace path")
    check("qwen2.5:14b  [loaded" in s and "GPU 100%" in s, "shows the loaded chat model with GPU %")
    check("qwen2.5-coder:14b  [not loaded]" in s, "shows the coder model as not loaded")
    check(f"voice        {j.tts.voice_key}" in s, "shows the active voice")
    j.set_active(True)
    check("AWAKE" in j.status_text(), "shows awake after waking")

    print("\n[6] Inactivity timer")
    j.idle_timeout = 300
    t0 = time.time()
    j.set_active(True)
    j.last_activity = t0 - 299
    check(j.check_inactivity(t0) is False and j.active, "299 s idle: still awake")
    j.last_activity = t0 - 301
    j.confirming.set()
    check(j.check_inactivity(t0) is False and j.active, "301 s idle but a confirmation is pending: timer paused")
    j.confirming.clear()
    j.turn_running.set()
    check(j.check_inactivity(t0) is False and j.active, "301 s idle but a request/coding turn is running: timer paused")
    j.turn_running.clear()
    said.clear()
    check(j.check_inactivity(t0) is True and not j.active, "301 s idle, nothing pending: goes back to sleep")
    check(said == ["Going quiet, sir."], f'says "Going quiet, sir." ({said})')
    j.idle_timeout = 0
    j.set_active(True)
    j.last_activity = t0 - 99999
    check(j.check_inactivity(t0) is False, "timeout 0 disables it")
    j.idle_timeout = 300

    print("\n[7] The idle clock restarts after a confirmation")
    j.set_active(True)
    j.last_activity = time.time() - 250
    j.events.put(("text", "y", "en"))
    check(j.confirm("Shall I run x.py?") is True, "typed 'y' approves")
    check(time.time() - j.last_activity < 5, "last_activity was refreshed when the question was answered")
    j.events.put(("voice", "yes", "en"))
    j.set_active(False)
    j.events.put(("text", "n", "en"))
    check(j.confirm("Delete x.py?") is False, "ambient voice 'yes' while asleep is ignored; typed 'n' declines")

    j.close()
    j.memory.close()
    for f in ("data/tj_settings.json", "data/tj_memory.db"):
        resolve(f).unlink(missing_ok=True)
    print("\nALL JARVIS BEHAVIOUR TESTS PASSED")


if __name__ == "__main__":
    main()
