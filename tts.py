"""Offline text-to-speech with Piper, selectable voices and an effects chain.

Speech is produced by two worker threads (synthesis -> playback) so the first sentence starts
playing while later ones are still being generated. stop() cuts speech instantly.

Standalone test:  python tts.py            (speaks a test line in every voice)
                  python tts.py --save     (also writes WAVs to data/tts_test/)
                  python tts.py --no-play  (only writes WAVs to data/tts_test/; silent)
"""
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import requests
import sounddevice as sd

from common import Settings, load_config, resolve, setup_logging
from voices import effects

log = logging.getLogger("jarvis.tts")

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


@dataclass
class Voice:
    key: str
    label: str
    model: str
    length_scale: float = 1.0
    effects: list = field(default_factory=list)
    aliases: list = field(default_factory=list)   # alternative spellings Whisper may produce


def _model_urls(model: str) -> tuple[str, str]:
    locale, name, quality = model.split("-", 2)
    lang = locale.split("_")[0]
    base = f"{HF_BASE}/{lang}/{locale}/{name}/{quality}/{model}"
    return base + ".onnx", base + ".onnx.json"


def _download(url: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        last = 0.0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and time.time() - last > 2:
                    log.info("  downloading %s: %d%%", dest.name, done * 100 // total)
                    last = time.time()
    tmp.replace(dest)


class TTS:
    def __init__(self, cfg: dict, settings: Settings):
        t = cfg["tts"]
        self.settings = settings
        self.piper_dir = resolve(t.get("piper_dir", "models/piper"))
        self.output_device = t.get("output_device")
        self.voices: dict[str, Voice] = {}
        for key, v in t["voices"].items():
            self.voices[key] = Voice(
                key=key, label=v.get("label", key), model=v["model"],
                length_scale=float(v.get("length_scale", 1.0)), effects=v.get("effects") or [],
                aliases=v.get("aliases") or [],
            )
        saved = settings.get("voice", t.get("default_voice"))
        self.voice_key = saved if saved in self.voices else next(iter(self.voices))
        log.info("Active voice: %s (model=%s, effects=%d)", self.voice_key, self.voice.model, len(self.voice.effects))
        self._muted = bool(settings.get("muted", False))

        self._piper = {}                       # model name -> PiperVoice
        self._piper_lock = threading.Lock()
        self._synth_q: queue.Queue = queue.Queue()
        self._play_q: queue.Queue = queue.Queue()
        self._gen = 0
        self._pending = 0
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._stop_play = threading.Event()
        self.last_activity = 0.0               # time speech last started/ended (echo guard)
        self._started = False

    # ---- state -------------------------------------------------------------------------
    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        self.settings.set("muted", muted)
        if muted:
            self.stop()

    @property
    def voice(self) -> Voice:
        return self.voices[self.voice_key]

    def set_voice(self, key: str) -> Voice:
        if key not in self.voices:
            raise KeyError(key)
        self._get_piper(self.voices[key].model)   # download/load FIRST: a failure must not leave a half-switched state
        self.voice_key = key
        self.settings.set("voice", key)
        return self.voice

    def next_voice(self) -> Voice:
        keys = list(self.voices)
        return self.set_voice(keys[(keys.index(self.voice_key) + 1) % len(keys)])

    @property
    def is_speaking(self) -> bool:
        return not self._idle.is_set()

    # ---- model loading -----------------------------------------------------------------
    def _get_piper(self, model: str):
        with self._piper_lock:
            if model in self._piper:
                return self._piper[model]
            from piper import PiperVoice
            onnx = self.piper_dir / f"{model}.onnx"
            conf = self.piper_dir / f"{model}.onnx.json"
            if not onnx.exists() or not conf.exists():
                log.info("Downloading Piper voice %s ...", model)
                u_onnx, u_conf = _model_urls(model)
                _download(u_conf, conf)
                _download(u_onnx, onnx)
            pv = PiperVoice.load(str(onnx), config_path=str(conf))
            self._piper[model] = pv
            return pv

    def preload(self) -> None:
        """Download/load the active voice (and fetch the others lazily when selected)."""
        self._get_piper(self.voice.model)

    # ---- synthesis ---------------------------------------------------------------------
    def synthesize(self, text: str, voice: Voice | None = None) -> tuple[np.ndarray, int]:
        voice = voice or self.voice
        pv = self._get_piper(voice.model)
        ls = voice.length_scale / effects.duration_factor(voice.effects)  # undo pitch-shift slow-down
        if hasattr(pv, "synthesize_stream_raw"):                          # piper-tts 1.2.x
            raw = b"".join(pv.synthesize_stream_raw(text, length_scale=ls))
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            sr = pv.config.sample_rate
        else:                                                              # piper-tts 1.3+
            from piper import SynthesisConfig
            chunks, sr = [], pv.config.sample_rate
            for c in pv.synthesize(text, syn_config=SynthesisConfig(length_scale=ls)):
                chunks.append(c.audio_float_array)
                sr = c.sample_rate
            audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        if len(audio) == 0:
            return audio, sr
        audio = effects.apply_chain(audio, sr, voice.effects) if voice.effects else audio
        return audio.astype(np.float32), sr

    # ---- playback pipeline -------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._synth_loop, name="tts-synth", daemon=True).start()
        threading.Thread(target=self._play_loop, name="tts-play", daemon=True).start()

    def say(self, text: str) -> None:
        """Queue text for speech (no-op while muted)."""
        text = text.strip()
        if not text or self._muted:
            return
        self.start()
        with self._lock:
            self._pending += 1
            self._idle.clear()
            self.last_activity = time.time()
            self._synth_q.put((self._gen, text, self.voice))

    def _finish_item(self) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)
            if self._pending == 0:
                self._idle.set()
            self.last_activity = time.time()

    def _synth_loop(self) -> None:
        while True:
            gen, text, voice = self._synth_q.get()
            if gen != self._gen:
                self._finish_item()
                continue
            try:
                audio, sr = self.synthesize(text, voice)
            except Exception as e:
                log.error("TTS synthesis failed: %s", e)
                self._finish_item()
                continue
            if gen != self._gen or len(audio) == 0:
                self._finish_item()
                continue
            self._play_q.put((gen, audio, sr))

    def _play_loop(self) -> None:
        while True:
            gen, audio, sr = self._play_q.get()
            try:
                if gen == self._gen:
                    self._play(audio, sr, gen)
            except Exception as e:
                log.error("Audio playback failed: %s", e)
            finally:
                self._finish_item()

    def _play(self, audio: np.ndarray, sr: int, gen: int) -> None:
        self._stop_play.clear()
        pad = np.zeros(int(sr * 0.08), np.float32)
        data = np.concatenate([audio, pad]).reshape(-1, 1)
        with sd.OutputStream(samplerate=sr, channels=1, dtype="float32", device=self.output_device) as out:
            step = 1024
            for i in range(0, len(data), step):
                if self._stop_play.is_set() or gen != self._gen:
                    break
                out.write(data[i:i + step])

    def stop(self) -> None:
        """Interrupt speech immediately and drop everything queued."""
        with self._lock:
            self._gen += 1
        self._stop_play.set()
        for q in (self._synth_q, self._play_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
                self._finish_item()

    def wait_idle(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)


def _self_test() -> None:
    setup_logging("INFO")
    cfg = load_config()
    settings = Settings(resolve("data/tts_test_settings.json"), {})
    tts = TTS(cfg, settings)
    no_play = "--no-play" in sys.argv          # synthesize + write WAVs only; nothing comes out of the speakers
    save = "--save" in sys.argv or no_play
    if save:
        import wave
        out_dir = resolve("data/tts_test")
        out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for key, v in tts.voices.items():
        line = f"Hello, this is the {v.label} voice. Systems are online, and the weather is nominal."
        log.info("Voice '%s' (%s)", key, v.model)
        try:
            tts.set_voice(key)
            if save:
                audio, sr = tts.synthesize(line, v)
                with wave.open(str(out_dir / f"{key}.wav"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(sr)
                    w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
            if not no_play:
                tts.say(line)
                tts.wait_idle(60)
        except Exception as e:
            failures += 1
            log.error("  FAILED: %s", e)
    log.info("Done. %d/%d voices OK.", len(tts.voices) - failures, len(tts.voices))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    _self_test()
