"""Speech input: microphone capture with an adaptive VAD, and faster-whisper transcription.

Microphone  -> pushes complete utterances (float32 16 kHz arrays) onto .utterances
STT         -> transcribe(audio) returns (text, language) with FR/EN auto-detection

Standalone test:  python stt.py          (synthesizes a sentence with Piper and transcribes it)
                  python stt.py --mic    (then listens to your microphone and prints what it hears)
"""
import collections
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from common import load_config, normalize, resolve, setup_logging

log = logging.getLogger("jarvis.stt")


class MicrophoneError(RuntimeError):
    pass


# ---- input device selection / diagnostics (used by audio.input_device and the /mic command) -------------------
def list_input_devices() -> list[dict]:
    """All input-capable devices: {index, name, hostapi, channels, default}."""
    import sounddevice as sd
    apis = sd.query_hostapis()
    try:
        default_idx = sd.default.device[0]
    except Exception:
        default_idx = -1
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append({"index": i, "name": d["name"], "hostapi": apis[d["hostapi"]]["name"],
                        "channels": d["max_input_channels"], "default": i == default_idx})
    return out


def resolve_input_device(spec) -> tuple[int | None, str]:
    """audio.input_device -> (device index or None for the system default, display name).
    spec: ""/None = default, an index (3 or "3"), or part of a name ("SoloCast", case-insensitive)."""
    import sounddevice as sd
    devices = list_input_devices()
    if spec is None or str(spec).strip().lower() in ("", "default", "none", "null"):
        try:
            return None, sd.query_devices(kind="input")["name"]
        except Exception as e:
            raise MicrophoneError(f"No default input device ({e}).") from e
    s = str(spec).strip()
    if s.isdigit():
        for d in devices:
            if d["index"] == int(s):
                return d["index"], d["name"]
        raise MicrophoneError(f"Input device index {s} does not exist or has no input. Run /mic to list devices.")
    matches = [d for d in devices if s.lower() in d["name"].lower()]
    if not matches:
        names = ", ".join(f"[{d['index']}] {d['name']}" for d in devices) or "none found"
        raise MicrophoneError(f"No input device matches '{s}'. Available: {names}")
    # The same physical mic appears once per host API (MME, DirectSound, WASAPI...): prefer the default API's entry.
    default_api = sd.query_hostapis(sd.default.hostapi)["name"] if sd.default.hostapi >= 0 else None
    matches.sort(key=lambda d: (d["hostapi"] != default_api, d["index"]))
    return matches[0]["index"], matches[0]["name"]


def diagnose_level(peak: float, rms: float) -> tuple[str, str]:
    """Interpret a level measurement -> (status, advice). status: silent | quiet | ok."""
    if peak < 1e-5:
        return "silent", (
            "The microphone delivers pure digital silence. Likely causes:\n"
            "  1. Hardware mute: a mute button / tap-to-mute sensor on the mic or headset (check its LED).\n"
            "  2. Windows privacy: Settings > Privacy & security > Microphone > turn on 'Microphone access' and\n"
            "     'Let desktop apps access your microphone'.\n"
            "  3. Wrong device: pick another from the list above and set audio.input_device in config.yaml\n"
            "     (e.g. \"SoloCast\"), or check Settings > System > Sound > Input that the device is enabled and its volume is not 0.")
    if peak < 0.01:
        return "quiet", (
            "Very little signal (peak below 0.01). If you were speaking, raise the input volume in Windows Sound settings,\n"
            "move closer to the mic, or check that this is the right device. In a quiet room this is normal.")
    return "ok", "Signal detected; the microphone is delivering audio."


def measure_level(device_spec, seconds: float = 2.0, live_mic=None) -> dict:
    """Measure the input level for `seconds`. Uses the running Microphone if given, else a short temporary stream."""
    if live_mic is not None:
        peak, rms = live_mic.measure(seconds)
        name = live_mic.device_name
    else:
        import sounddevice as sd
        idx, name = resolve_input_device(device_spec)
        try:
            rec = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype="float32", device=idx)
            sd.wait()
        except Exception as e:
            raise MicrophoneError(f"Could not open '{name}': {e}") from e
        x = rec[:, 0]
        peak, rms = float(np.max(np.abs(x))), float(np.sqrt(np.mean(x ** 2)))
    status, advice = diagnose_level(peak, rms)
    return {"device": name, "peak": peak, "rms": rms, "status": status, "advice": advice}


def _setup_cuda_dlls() -> None:
    """Make the pip-installed NVIDIA cuBLAS/cuDNN DLLs visible to CTranslate2 on Windows."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if os.name != "nt":
        return
    import site
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    for root in roots:
        nvidia = Path(root) / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


# Phrases Whisper invents on silence/noise.
_HALLUCINATIONS = {
    "thank you", "thanks", "thank you for watching", "thanks for watching", "bye", "you",
    "merci", "merci d avoir regarde cette video", "sous titres realises par la communaute d amara org",
    "sous titrage st 501", "subtitles by the amara org community", "amara org",
}


class STT:
    def __init__(self, cfg: dict):
        s = cfg["stt"]
        self.allowed = list(s.get("allowed_languages", ["en", "fr"]))
        self.beam = int(s.get("beam_size", 3))
        self.prompt = s.get("initial_prompt") or None
        self.model = None
        self.device = s.get("device", "cuda")
        self._load(s)

    def _load(self, s: dict) -> None:
        from faster_whisper import WhisperModel
        _setup_cuda_dlls()
        name = s.get("model", "medium")
        model_dir = str(resolve(s.get("model_dir", "models/whisper")))
        attempts = [(self.device, s.get("compute_type", "float16"))]
        if self.device == "cuda":
            attempts.append(("cpu", "int8"))
        last_err = None
        for device, compute in attempts:
            try:
                log.info("Loading Whisper '%s' on %s (%s)... first run downloads the model.", name, device, compute)
                model = WhisperModel(name, device=device, compute_type=compute, download_root=model_dir)
                # Warm-up: CUDA library problems surface on first inference, not at load.
                segs, _ = model.transcribe(np.zeros(16000, np.float32), beam_size=1)
                list(segs)
                self.model, self.device = model, device
                log.info("Whisper ready on %s.", device)
                return
            except Exception as e:
                last_err = e
                log.warning("Whisper failed on %s: %s", device, e)
        raise RuntimeError(f"Could not load Whisper model '{name}': {last_err}")

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """Return (text, language_code). Text is '' if nothing intelligible was said."""
        kwargs = dict(
            beam_size=self.beam, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            initial_prompt=self.prompt, condition_on_previous_text=False,
            temperature=0.0, no_speech_threshold=0.6,
        )
        segments, info = self.model.transcribe(audio, language=None, **kwargs)
        lang = info.language
        if lang not in self.allowed:
            probs = dict(info.all_language_probs or [])
            lang = max(self.allowed, key=lambda l: probs.get(l, 0.0)) if probs else self.allowed[0]
            segments, info = self.model.transcribe(audio, language=lang, **kwargs)
        parts = []
        for seg in segments:
            if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                continue
            parts.append(seg.text.strip())
        text = " ".join(p for p in parts if p).strip()
        if normalize(text) in _HALLUCINATIONS:
            return "", lang
        return text, lang


class Microphone:
    """Continuously listens, segments speech with an adaptive energy VAD, and emits utterances.

    While Jarvis is speaking, a much higher threshold applies (so he does not hear himself);
    crossing it triggers on_barge_in() and the utterance is captured as normal.
    """

    def __init__(self, cfg: dict, is_speaking=lambda: False, last_speech_time=lambda: 0.0, on_barge_in=lambda: None):
        a = cfg["audio"]
        self.sr = int(a.get("sample_rate", 16000))
        self.block = int(self.sr * int(a.get("block_ms", 32)) / 1000)
        self.block_ms = self.block * 1000 / self.sr
        self.device_spec = a.get("input_device")
        self.device_index: int | None = None
        self.device_name = ""
        self._peak = 0.0                 # running stats for measure() (/mic)
        self._sumsq = 0.0
        self._n = 0
        self.ratio = float(a.get("vad_ratio", 3.0))
        self.min_rms = float(a.get("vad_min_rms", 0.008))
        self.start_blocks = int(a.get("start_blocks", 3))
        self.silence_ms = int(a.get("silence_ms", 800))
        self.min_speech_ms = int(a.get("min_speech_ms", 300))
        self.max_blocks = int(a.get("max_utterance_s", 30) * 1000 / self.block_ms)
        self.preroll_blocks = max(1, int(a.get("preroll_ms", 320) / self.block_ms))
        self.barge_in = bool(a.get("barge_in", True))
        self.barge_ratio = float(a.get("barge_in_ratio", 4.0))
        self.barge_blocks = int(a.get("barge_in_blocks", 8))
        self.is_speaking = is_speaking
        self.last_speech_time = last_speech_time
        self.on_barge_in = on_barge_in

        self.utterances: queue.Queue = queue.Queue()
        self.level = 0.0          # latest RMS (for diagnostics)
        self._blocks: queue.Queue = queue.Queue(maxsize=500)
        self._stream = None
        self._stop = threading.Event()
        self._paused = threading.Event()

    # ---- lifecycle ---------------------------------------------------------------------
    def start(self) -> None:
        import sounddevice as sd
        self.device_index, self.device_name = resolve_input_device(self.device_spec)   # MicrophoneError if no match
        try:
            if self.device_index is None:
                sd.check_input_settings(samplerate=self.sr, channels=1)
            self._stream = sd.InputStream(
                samplerate=self.sr, channels=1, dtype="float32", blocksize=self.block,
                device=self.device_index, callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            raise MicrophoneError(
                f"No usable microphone ({e}). Run /mic to list input devices, check Windows Settings > System > "
                "Sound > Input, and make sure microphone access is allowed for desktop apps."
            ) from e
        log.info("Microphone: %s%s", self.device_name,
                 f" [index {self.device_index}]" if self.device_index is not None else " (system default)")
        threading.Thread(target=self._process_loop, name="mic-vad", daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

    def measure(self, seconds: float = 2.0) -> tuple[float, float]:
        """(peak, rms) of the live input over the next `seconds`."""
        self._peak, self._sumsq, self._n = 0.0, 0.0, 0
        time.sleep(seconds)
        n = max(self._n, 1)
        return self._peak, (self._sumsq / n) ** 0.5

    def _callback(self, indata, frames, time_info, status) -> None:
        try:
            self._blocks.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # drop audio rather than block the audio thread

    # ---- VAD ---------------------------------------------------------------------------
    def _process_loop(self) -> None:
        noise = self.min_rms / 2
        calib = int(500 / self.block_ms)
        preroll = collections.deque(maxlen=self.preroll_blocks)
        buf: list = []
        in_speech = False
        loud_run = 0
        silent = 0
        started_by_barge = False
        while not self._stop.is_set():
            try:
                blk = self._blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            rms = float(np.sqrt(np.mean(blk ** 2)) + 1e-9)
            self.level = rms
            self._peak = max(self._peak, float(np.max(np.abs(blk))))
            self._sumsq += rms * rms
            self._n += 1
            now = time.time()
            echo = self.is_speaking() or (now - self.last_speech_time() < 0.5)
            thr = max(noise * self.ratio, self.min_rms) * (self.barge_ratio if echo else 1.0)

            if not in_speech:
                preroll.append(blk)
                if calib > 0:                       # initial noise-floor calibration
                    noise = max(noise * 0.9 + rms * 0.1, 1e-4)
                    calib -= 1
                    continue
                if not echo:
                    # follow the room noise: fast down, slow up
                    noise = noise * 0.9 + rms * 0.1 if rms < noise else noise * 0.995 + rms * 0.005
                if echo and not self.barge_in:
                    loud_run = 0
                    continue
                loud_run = loud_run + 1 if rms > thr else 0
                need = self.barge_blocks if echo else self.start_blocks
                if loud_run >= need:
                    in_speech, silent = True, 0
                    started_by_barge = echo
                    buf = list(preroll)[-need:] if echo else list(preroll)
                    preroll.clear()
                    loud_run = 0
                    if echo:
                        log.info("Barge-in: interrupting speech.")
                        try:
                            self.on_barge_in()
                        except Exception:
                            log.exception("barge-in handler failed")
            else:
                buf.append(blk)
                keep = max(noise * self.ratio, self.min_rms) * 0.7
                silent = 0 if rms > keep else silent + 1
                if silent * self.block_ms >= self.silence_ms or len(buf) >= self.max_blocks:
                    trail = int(200 / self.block_ms)
                    audio_blocks = buf[: len(buf) - max(0, silent - trail)]
                    speech_ms = (len(buf) - silent) * self.block_ms
                    if speech_ms >= self.min_speech_ms:
                        self.utterances.put(np.concatenate(audio_blocks).astype(np.float32))
                    in_speech, buf, silent = False, [], 0
                    started_by_barge = False
                    preroll.clear()

    def drain(self) -> None:
        while True:
            try:
                self.utterances.get_nowait()
            except queue.Empty:
                break


def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, 16000)
    return resample_poly(audio, 16000 // g, sr // g).astype(np.float32)


def _self_test() -> None:
    setup_logging("INFO")
    cfg = load_config()
    from common import Settings
    from tts import TTS
    tts = TTS(cfg, Settings(resolve("data/tts_test_settings.json"), {}))
    stt = STT(cfg)
    ok = True
    for text, expect in [("Hello Jarvis, what is the weather like today?", "jarvis")]:
        audio, sr = tts.synthesize(text, tts.voices[next(iter(tts.voices))])
        t0 = time.time()
        got, lang = stt.transcribe(_resample_to_16k(audio, sr))
        log.info("TTS->STT: %r -> %r [%s] in %.2fs", text, got, lang, time.time() - t0)
        ok &= expect in normalize(got)
    log.info("STT self-test %s", "PASSED" if ok else "FAILED")

    if "--mic" in sys.argv:
        mic = Microphone(cfg)
        try:
            mic.start()
        except MicrophoneError as e:
            log.error("%s", e)
            sys.exit(1)
        log.info("Listening... speak (French or English). Ctrl+C to quit.")
        try:
            while True:
                try:
                    audio = mic.utterances.get(timeout=0.5)
                except queue.Empty:
                    continue
                text, lang = stt.transcribe(audio)
                log.info("[%s] %s  (%.1fs of audio)", lang, text or "<nothing>", len(audio) / 16000)
        except KeyboardInterrupt:
            mic.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _self_test()
