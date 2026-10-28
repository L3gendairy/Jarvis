"""Full voice path without a microphone or speakers:

  TTS-generated WAV -> padded with room noise -> Microphone VAD (utterance segmentation) -> faster-whisper STT
  -> Jarvis.handle: strict wake-word rule -> command router -> tts.set_voice -> settings file

Run:  python test_voice_pipeline.py
SILENT: TTS output is muted; the input WAVs are written to data/tts_test/pipeline_*.wav. The microphone is not used.
No LLM is needed (every phrase here is handled before the LLM is reached).
"""
import logging
import queue
import threading
import time
import wave

import numpy as np

from common import Settings, load_config, resolve, setup_logging
from main import Jarvis
from stt import STT, Microphone, _resample_to_16k

SETTINGS = "data/vp_test_settings.json"
DB = "data/vp_test_memory.db"


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise SystemExit(1)


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def save_wav16(path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def through_vad(cfg: dict, audio16k: np.ndarray, rng) -> list[np.ndarray]:
    """Feed audio through the real Microphone VAD loop exactly as the sound card would (no device opened)."""
    noise = lambda s: (rng.standard_normal(int(16000 * s)) * 0.002).astype(np.float32)
    stream = np.concatenate([noise(1.5), audio16k, noise(1.6)])
    mic = Microphone(cfg)
    t = threading.Thread(target=mic._process_loop, daemon=True)
    t.start()
    for i in range(0, len(stream) - mic.block + 1, mic.block):
        mic._blocks.put(stream[i:i + mic.block])
    while not mic._blocks.empty():
        time.sleep(0.05)
    time.sleep(0.4)
    mic._stop.set()
    out = []
    while True:
        try:
            out.append(mic.utterances.get_nowait())
        except queue.Empty:
            return out


def main() -> None:
    setup_logging("INFO")
    for f in (SETTINGS, DB):
        resolve(f).unlink(missing_ok=True)
    cfg = load_config()
    cfg["settings_file"] = SETTINGS
    cfg["memory"]["db_path"] = DB
    j = Jarvis(cfg, text_only=True)
    j.tts._muted = True                                   # nothing reaches the speakers
    stt = STT(cfg)
    keys = list(j.tts.voices)
    cap = Capture()
    logging.getLogger("jarvis").addHandler(cap)
    rng = np.random.default_rng(3)
    out_dir = resolve("data/tts_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    speaker = j.tts.voices["ryan"]                        # the "user" speaking in the WAVs

    def say_to_jarvis(phrase: str, tag: str) -> str:
        audio, sr = j.tts.synthesize(phrase, speaker)
        a16 = (_resample_to_16k(audio, sr) * 0.3).astype(np.float32)
        save_wav16(out_dir / f"pipeline_{tag}.wav", a16)
        utts = through_vad(cfg, a16, rng)
        check(len(utts) == 1, f"VAD found exactly one utterance in '{phrase}' ({len(utts)})")
        text, lang = stt.transcribe(utts[0])
        print(f"        Whisper heard [{lang}]: {text!r}")
        j.handle("voice", text, lang)
        return text

    print("\n[1] 'Jarvis, change voice.'  (asleep -> wake word -> cycle)")
    check(not j.active and j.tts.voice_key == keys[0], f"start: asleep, voice '{keys[0]}'")
    say_to_jarvis("Jarvis, change voice.", "change_voice")
    check(j.active, "wake word recognised and Jarvis is now active")
    check(j.tts.voice_key == keys[1], f"voice cycled to '{j.tts.voice_key}'")

    print("\n[2] 'Jarvis, use the cyn voice.'  (active -> leading wake word stripped -> select Cyn)")
    say_to_jarvis("Jarvis, use the cyn voice.", "use_cyn")
    check(j.tts.voice_key == "cyn", f"voice is '{j.tts.voice_key}'")
    check(Settings(resolve(SETTINGS), {}).get("voice") == "cyn", "choice saved in the settings file")
    check(any("active voice now: cyn" in m for m in cap.messages), "debug line logged: command + active voice")

    print("\n[3] Strict wake word while sleeping")
    j.set_active(False)
    say_to_jarvis("I told Jarvis to turn the lights off in the kitchen.", "false_wake")
    check(not j.active, "mid-sentence 'Jarvis' in a long sentence did NOT wake him")

    print("\n[4] Voice-like phrase that matches no command is logged")
    cap.messages.clear()
    say_to_jarvis("What a lovely voice that singer has.", "voice_like")
    check(any(m.startswith("Voice-like phrase not matched: '") for m in cap.messages),
          "logged: " + next((m for m in cap.messages if m.startswith("Voice-like")), "<missing>"))
    check(not j.active, "and it did not wake him")

    j.close()
    j.memory.close()
    for f in (SETTINGS, DB):
        resolve(f).unlink(missing_ok=True)
    print("\nALL VOICE PIPELINE TESTS PASSED  (WAVs in data/tts_test/pipeline_*.wav)")


if __name__ == "__main__":
    main()
