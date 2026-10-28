"""Voice switching test through the real Jarvis.handle path (command router -> tts.set_voice -> settings file).

Run:  python test_voice_switching.py
SILENT: TTS output is muted (nothing reaches the speakers) and each voice's test line is written to
data/tts_test/voice_<name>.wav instead. The microphone is not used.
"""
import subprocess
import sys
import wave

import numpy as np

from common import Settings, load_config, resolve, setup_logging
from main import Jarvis

SETTINGS = "data/vs_test_settings.json"
DB = "data/vs_test_memory.db"


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise SystemExit(1)


def write_wav(path, audio: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def main() -> None:
    setup_logging("INFO")
    for f in (SETTINGS, DB):
        resolve(f).unlink(missing_ok=True)
    cfg = load_config()
    cfg["settings_file"] = SETTINGS
    cfg["memory"]["db_path"] = DB
    j = Jarvis(cfg, text_only=True)
    j.tts._muted = True                                   # silent: say() returns before queueing any audio
    keys = list(j.tts.voices)
    out_dir = resolve("data/tts_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    said = []                                             # (text, voice active at the moment it was said)
    orig_say = j.tts.say
    j.tts.say = lambda t: (said.append((t, j.tts.voice_key)), orig_say(t))[0]

    print("\n[1] Typed and spoken commands through Jarvis.handle (wake word + command detection)")
    j.handle("text", "list voices", "en")
    check("One, Jarvis. Two, Ryan. Three, Amy. Four, Cyn." in said[-1][0], "'list voices' speaks the numbered list")

    j.handle("voice", "Jarvis, change voice.", "en")      # asleep -> wakes him, then cycles
    check(j.active, "wake word activated Jarvis and was stripped from the command")
    check(j.tts.voice_key == keys[1], f"'change voice' cycled to {keys[1]}")
    check(said[-1] == (f"Voice changed to {keys[1].title()}.", keys[1]),
          f"confirmation is spoken in the NEW voice: {said[-1]}")

    for source, phrase, expect in [
        ("text", "use the cyn voice", "cyn"),
        ("voice", "Use cyn voice.", "cyn"),
        ("voice", "Jarvis, switch to the Cyn voice", "cyn"),
        ("voice", "sin voice", "cyn"),
        ("voice", "use the sinn voice", "cyn"),
        ("text", "voice 2", keys[1]),
        ("text", "voice three", keys[2]),
    ]:
        j.tts.set_voice(keys[0])
        j.handle(source, phrase, "en")
        check(j.tts.voice_key == expect, f"{source}: {phrase!r} -> {j.tts.voice_key}")

    before = j.tts.voice_key
    j.handle("text", "use the foo voice", "en")
    check(j.tts.voice_key == before and "I don't have a voice called foo" in said[-1][0],
          "unknown voice 'foo' leaves the voice unchanged and says so")

    print("\n[2] Every voice: switch by command, render a line in the NEW voice to a WAV (no playback)")
    for k in keys:
        j.handle("text", f"use the {k} voice", "en")
        check(j.tts.voice_key == k and j.settings.get("voice") == k, f"active voice and settings are '{k}'")
        audio, sr = j.tts.synthesize(f"Voice changed to {k}.", j.tts.voice)
        write_wav(out_dir / f"voice_{k}.wav", audio, sr)
        check(len(audio) > 1000, f"rendered {len(audio) / sr:.1f}s of audio -> data/tts_test/voice_{k}.wav")

    print("\n[3] Cyn effect chain is applied")
    cyn, amy = j.tts.voices["cyn"], j.tts.voices["amy"]
    check(len(cyn.effects) > 0, f"cyn has {len(cyn.effects)} effects configured")
    a_fx, _ = j.tts.synthesize("Test line.", cyn)
    a_raw, _ = j.tts.synthesize("Test line.", amy)
    check(len(a_fx) != len(a_raw) or not np.allclose(a_fx[:1000], a_raw[:1000]), "cyn audio differs from the clean base voice")

    print("\n[4] Persistence after restart")
    j.handle("text", "use the cyn voice", "en")
    on_disk = Settings(resolve(SETTINGS), {}).get("voice")
    check(on_disk == "cyn", f"settings file on disk says '{on_disk}'")
    code = (
        "import sys; sys.path.insert(0, '.');"
        "from common import *; from tts import TTS;"
        f"cfg = load_config(); cfg['settings_file'] = '{SETTINGS}';"
        f"t = TTS(cfg, Settings(resolve('{SETTINGS}'), {{'voice': cfg['tts']['default_voice']}}));"
        "print('RESTARTED_VOICE=' + t.voice_key)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(resolve("."))).stdout
    restarted = next((l.split("=", 1)[1] for l in out.splitlines() if l.startswith("RESTARTED_VOICE=")), None)
    check(restarted == "cyn", f"fresh process starts with '{restarted}'")

    j.close()
    j.memory.close()
    for f in (SETTINGS, DB):
        resolve(f).unlink(missing_ok=True)
    print("\nALL VOICE SWITCHING TESTS PASSED")


if __name__ == "__main__":
    main()
