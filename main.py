"""Jarvis - local voice assistant. Run:  python main.py   (add --text to disable the microphone)

States:  SLEEPING  - hears everything, reacts only to the wake word "Jarvis" (at the start of what you say)
         ACTIVE    - every sentence is a command until you say "Jarvis stop" (or 5 idle minutes pass)
Typed input in this console always works, in either state.
"""
import argparse
import logging
import queue
import random
import re
import sys
import threading
import time

import hardware
import tools
from commands import Commands
from common import Settings, load_config, normalize, parse_yes_no, resolve, setup_logging
from llm import LLM, LLMError
from memory import Memory
from stt import STT, Microphone, MicrophoneError, list_input_devices, measure_level, resolve_input_device
from tools.code import CodeTools
from tts import TTS
from wakeword import WakeWord

log = logging.getLogger("jarvis")

HELP = """\
Voice / typed commands (say "Jarvis" first when sleeping):
  Jarvis stop               go back to sleep (wake word needed again; also automatic after 5 idle minutes)
  change voice / list voices / use the cyn voice / voice 2
  mute / unmute             mute = text-only replies
  remember that ...         forget ...        what do you know about me?
Coding (uses the coder model; files live in the ./workspace folder, scripts run in workspace/.venv):
  write a script that ...   code me ...       fix this code       make a program that ...
  open workspace            opens the workspace folder in Explorer
  list files                lists what is in the workspace
  run [file]                runs a script (asks for confirmation, shows the code first, warns about risky code)
  show [file]               prints a file's code in the console
  Running a script, deleting a file or installing a package always asks first: say yes / no, or type y / n.
Console only:
  /help                     this text
  /mic                      list input devices, show the one in use, measure 2 s of level (diagnoses a silent mic)
  /status                   loaded models + GPU %, voice, mute, awake/asleep, workspace
  /reset                    clear chat history
  /quit                     exit
"""

_VOICE_LIKE = re.compile(r"\bvoi(?:ce|x)s?\b")


class SpeechFilter:
    """Keeps code out of the speakers: drops ``` fenced blocks and strips markdown symbols."""

    def __init__(self):
        self.in_code = False

    def clean(self, sentence: str) -> str | None:
        """Return the text to speak, or None if this sentence is (part of) code."""
        fences = sentence.count("```")
        if self.in_code:
            if fences % 2 == 1:
                self.in_code = False
            return None
        if fences:
            before = sentence.split("```")[0]
            if fences % 2 == 1:
                self.in_code = True
            sentence = before
        sentence = re.sub(r"[`*#]+", "", sentence).strip()
        return sentence or None


class ConsoleUI:
    """What the code tools use to talk to the user: code listings, warnings, run output, yes/no questions."""

    def __init__(self, jarvis: "Jarvis"):
        self.j = jarvis

    def confirm(self, question: str) -> bool:
        return self.j.confirm(question)

    def show_code(self, rel: str, code: str) -> None:
        lines = code.splitlines() or [""]
        width = len(str(len(lines)))
        bar = "-" * 70
        print(f"\n{bar}\n {rel}  ({len(lines)} lines)\n{bar}")
        for i, line in enumerate(lines, 1):
            print(f"{i:>{width}} | {line}")
        print(f"{bar}\n")

    def show_warning(self, rel: str, findings: list[str]) -> None:
        bar = "!" * 70
        print(f"{bar}\n WARNING: {rel} does risky things:")
        for f in findings:
            print(f"   - {f}")
        print(f" (static scan: best effort, it cannot see everything. Read the code above before approving.)\n{bar}\n")

    def show_output(self, result) -> None:
        bar = "-" * 70
        print(f"\n{bar}\n result: {result.status}  (exit code {result.exit_code})\n{bar}")
        if result.stdout.strip():
            print(result.stdout.rstrip())
        if result.stderr.strip():
            print("[stderr]\n" + result.stderr.rstrip())
        print(f"{bar}\n")


class Jarvis:
    def __init__(self, cfg: dict, text_only: bool):
        self.cfg = cfg
        self.text_only = text_only
        self.settings = Settings(resolve(cfg.get("settings_file", "data/settings.json")),
                                 {"voice": cfg["tts"].get("default_voice"), "muted": False})
        self.events: queue.Queue = queue.Queue()
        self.active = False
        self.last_activity = time.time()
        self.idle_timeout = float(cfg["wake_word"].get("inactivity_timeout_s", 0) or 0)
        self.interrupt = threading.Event()
        self.turn_running = threading.Event()
        self.confirming = threading.Event()
        self.running = True
        self.mic: Microphone | None = None
        self.stt: STT | None = None

        self.wake = WakeWord(cfg)
        self.memory = Memory(resolve(cfg["memory"]["db_path"]))
        self.tts = TTS(cfg, self.settings)
        self.llm = LLM(cfg)
        self.code = CodeTools(cfg, ConsoleUI(self))
        tools.configure(self.code)
        self.commands = Commands(self.tts, self.memory, self.wake, self.code, on_code_used=self.llm.mark_coding)

    # ---- startup -----------------------------------------------------------------------
    def setup(self) -> None:
        log.info("Checking Ollama...")
        self.llm.ensure_server()          # raises LLMError with a clear message
        self.llm.ensure_models()
        log.info("LLM ready: chat=%s, coder=%s, workspace=%s", self.llm.model, self.llm.coder_model, self.code.root)

        log.info("Loading voice '%s'...", self.tts.voice_key)
        try:
            self.tts.preload()
            self.tts.start()
        except Exception as e:
            log.error("Could not load the TTS voice (%s). Falling back to text-only replies.", e)
            self.tts.set_muted(True)

        if self.text_only:
            log.info("Text-only mode: microphone disabled.")
        else:
            self._setup_listening()

        # Last: load the chat model AFTER Whisper so the GPU/CPU split reflects what is really left for it.
        try:
            log.info("Loading chat model into memory...")
            self.llm.preload()
            self.llm.gpu_report()
        except Exception as e:
            log.warning("Could not check GPU placement: %s", e)

    def _setup_listening(self) -> None:
        try:
            self.stt = STT(self.cfg)
        except Exception as e:
            log.error("Speech recognition unavailable: %s", e)
            log.error("Continuing with typed input only.")
            return
        try:
            self.mic = Microphone(
                self.cfg,
                is_speaking=lambda: self.tts.is_speaking,
                last_speech_time=lambda: self.tts.last_activity,
                on_barge_in=self.on_barge_in,
            )
            self.mic.start()
        except MicrophoneError as e:
            log.error("%s", e)
            log.error("Continuing with typed input only. Use /mic to list input devices.")
            self.mic = None

    def on_barge_in(self) -> None:
        self.tts.stop()
        # While waiting for a yes/no the user is expected to talk; that must not abort the request.
        if self.turn_running.is_set() and not self.confirming.is_set():
            self.interrupt.set()

    # ---- input threads -----------------------------------------------------------------
    def _stdin_loop(self) -> None:
        while self.running:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:                       # EOF
                return
            line = line.strip().lstrip("﻿")
            if line:
                self.events.put(("text", line, "en"))

    def _stt_loop(self) -> None:
        while self.running:
            try:
                audio = self.mic.utterances.get(timeout=0.5)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                text, lang = self.stt.transcribe(audio)
            except Exception as e:
                log.error("Transcription failed: %s", e)
                continue
            if text:
                log.info("Heard [%s] (%.1fs audio, %.2fs to transcribe): %s", lang, len(audio) / 16000, time.time() - t0, text)
                self.events.put(("voice", text, lang))

    # ---- output ------------------------------------------------------------------------
    def say(self, text: str) -> None:
        if not text:
            return
        log.info("Jarvis: %s%s", text, "  [muted]" if self.tts.muted else "")
        self.tts.say(text)

    def set_active(self, active: bool) -> None:
        if active != self.active:
            self.active = active
            log.info("== %s ==", "ACTIVE - listening (say 'Jarvis stop' to sleep)" if active
                     else "SLEEPING - waiting for 'Jarvis'")
        self.last_activity = time.time()

    def check_inactivity(self, now: float | None = None) -> bool:
        """Go back to sleep after `inactivity_timeout_s` without a command. The timer is paused while a yes/no
        question is pending, a request (coding turns included) is running, or he is speaking."""
        if not self.idle_timeout or not self.active:
            return False
        if self.confirming.is_set() or self.turn_running.is_set() or self.tts.is_speaking:
            return False
        now = time.time() if now is None else now
        if now - self.last_activity < self.idle_timeout:
            return False
        log.info("No command for %d s.", self.idle_timeout)
        self.set_active(False)
        self.say("Going quiet, sir.")
        return True

    def _log_voice_like(self, text: str) -> None:
        """Show what Whisper heard when it sounded like a voice command but nothing matched."""
        if _VOICE_LIKE.search(normalize(text)):
            log.info("Voice-like phrase not matched: '%s'", text)

    # ---- confirmation (run script / delete file / pip install) --------------------------
    def confirm(self, question: str) -> bool:
        """Ask a yes/no question and wait for a spoken yes/no or typed y/n. Silence or anything unclear = no."""
        timeout = float(self.cfg.get("code", {}).get("confirm_timeout_s", 90))
        self.confirming.set()
        held: list[tuple] = []
        try:
            log.info("CONFIRM: %s  [say yes / no, or type y / n]", question)
            self.tts.say(question)
            deadline = time.time() + timeout
            unclear = 0
            while time.time() < deadline and self.running:
                try:
                    event = self.events.get(timeout=0.5)
                except queue.Empty:
                    continue
                source, text, _lang = event
                if text.startswith("/"):
                    held.append(event)
                    continue
                if source == "voice" and not self.active:
                    continue                   # ambient speech must not approve running code
                answer = parse_yes_no(self.wake.strip_leading(text))
                if answer is None:
                    unclear += 1
                    if unclear >= 2:
                        log.info("Not a clear yes - taking that as a no.")
                        held.append(event)     # process it normally afterwards
                        return False
                    self.say("Sorry, yes or no?")
                    continue
                log.info("Answer: %s", "yes" if answer else "no")
                return answer
            log.info("No answer within %ds - taking that as a no.", timeout)
            return False
        finally:
            self.confirming.clear()
            self.last_activity = time.time()   # the inactivity clock restarts after the question
            for e in held:
                self.events.put(e)

    # ---- event handling ----------------------------------------------------------------
    def handle(self, source: str, text: str, lang: str) -> None:
        if source == "voice":
            if not self.active:
                found, rest = self.wake.find_for_wake(text)
                if not found:
                    log.info("(ignored - no wake word at the start)")
                    self._log_voice_like(text)
                    return
                if not rest or self.wake.is_stop(rest):
                    if rest:
                        return                 # "Jarvis stop" while already asleep
                    self.set_active(True)
                    self.say(random.choice(["Yes?", "At your service.", "I'm here.", "Listening."]))
                    return
                self.set_active(True)
                text = rest
            else:
                text = self.wake.strip_leading(text)
                if not text:
                    self.say(random.choice(["Yes?", "I'm here."]))
                    return
        else:
            text = self.wake.strip_leading(text)
            self.tts.stop()                    # typing interrupts speech

        self.last_activity = time.time()
        self.code.reset_turn()
        result = self.commands.handle(text)
        if result is not None:
            log.info("Command: %s", text)
            if result.display:
                print(result.display)
            self.say(result.reply)
            if result.reply:
                self.llm.add_exchange_note(text, result.reply)
            if result.action == "sleep":
                self.set_active(False)
            elif result.action == "quit":
                self.tts.wait_idle(8)
                self.running = False
            elif result.llm_prompt:
                self.ask_llm(result.llm_prompt, "en", force_coding=True)
            return
        self._log_voice_like(text)
        self.ask_llm(text, lang)

    def ask_llm(self, text: str, lang: str, force_coding: bool = False) -> None:
        coding = force_coding or self.llm.wants_coder(text)
        self.code.reset_turn(text)             # fresh 3-attempt budget; notes whether this request asks to run code
        facts = [] if coding else self.memory.relevant(text, self.cfg["memory"].get("max_injected", 8))
        if facts:
            log.info("Memory injected: %d fact(s)", len(facts))
        prompt = text if lang == "en" else f"{text}\n\n(The user spoke {lang.upper()}; reply in English.)"
        self.interrupt.clear()
        self.turn_running.set()
        speech = SpeechFilter()

        def on_sentence(s: str) -> None:
            if self.interrupt.is_set():
                return
            spoken = speech.clean(s)
            if spoken is None:                 # code: show it in the console, never speak it
                print(s)
                return
            log.info("Jarvis: %s%s", spoken, "  [muted]" if self.tts.muted else "")
            self.tts.say(spoken)

        try:
            if coding:
                log.info("Coding request -> %s", self.llm.coder_model)
            log.info("Thinking...")
            reply = self.llm.chat(prompt, facts, on_sentence=on_sentence,
                                  on_status=lambda s: log.info("Doing: %s", s),
                                  should_stop=self.interrupt.is_set, coding=coding,
                                  nudge_hook=self.code.nudge if coding else (lambda: None))
        except LLMError as e:
            self.turn_running.clear()
            log.error("%s", e)
            self.say("I can't reach my language model right now. Check the console for details.")
            return
        finally:
            self.turn_running.clear()
            self.last_activity = time.time()   # inactivity clock starts after the turn, not before it

        if self.interrupt.is_set():
            log.info("(interrupted)")
            return
        if self.cfg["memory"].get("extract_facts", True) and reply and not coding:
            threading.Thread(target=self._extract, args=(text, reply), daemon=True).start()

    def _extract(self, user_text: str, reply: str) -> None:
        try:
            added = self.memory.extract_and_store(self.llm, user_text, reply)
            for fact in added:
                log.info("Memory saved: %s", fact)
        except Exception as e:
            log.debug("fact extraction failed: %s", e)

    # ---- console commands --------------------------------------------------------------
    def cmd_mic(self) -> None:
        """/mic: list input devices, show the one in use, measure 2 s of level."""
        spec = self.cfg["audio"].get("input_device")
        try:
            devices = list_input_devices()
            if self.mic is not None:
                used_idx, used_name = self.mic.device_index, self.mic.device_name
            else:
                used_idx, used_name = resolve_input_device(spec)
        except MicrophoneError as e:
            print(f"Microphone problem: {e}")
            return
        print(f"\nInput devices (audio.input_device = {spec!r}):")
        for d in devices:
            in_use = (used_idx == d["index"]) or (used_idx is None and d["default"])
            tag = "  <- IN USE" if in_use else ("  (system default)" if d["default"] else "")
            print(f"  [{d['index']:>2}] {d['name']}  | {d['hostapi']} | {d['channels']} ch{tag}")
        print(f"\nUsing: {used_name}" + (f" [index {used_idx}]" if used_idx is not None else " (system default)"))
        if self.text_only:
            print("Text-only mode (--text): the microphone is disabled, so no level was measured.")
            return
        print("Measuring the input level for 2 seconds - say something...")
        try:
            res = measure_level(spec, 2.0, live_mic=self.mic)
        except MicrophoneError as e:
            print(f"Could not measure: {e}")
            return
        print(f"Level: peak {res['peak']:.4f}, rms {res['rms']:.5f}  ->  {res['status'].upper()}")
        print(res["advice"] + "\n")

    def status_text(self) -> str:
        """/status: loaded models + GPU %, voice, mute, awake/asleep, workspace."""
        try:
            loaded = {m["name"]: m for m in self.llm.loaded_models()}
            ps_error = ""
        except Exception as e:
            loaded, ps_error = {}, f" (could not query Ollama: {e})"

        def model_line(label: str, name: str) -> str:
            m = loaded.get(name) or loaded.get(name + ":latest") or loaded.get(name.removesuffix(":latest"))
            if m is None:
                return f"  {label:<12} {name}  [not loaded]"
            return f"  {label:<12} {name}  [loaded, {m['size'] / 1e9:.1f} GB, GPU {m['gpu_pct']}% / CPU {100 - m['gpu_pct']}%]"

        v = self.tts.voice
        lines = ["", "Jarvis status" + ps_error,
                 f"  state        {'AWAKE (active)' if self.active else 'ASLEEP (say Jarvis)'}",
                 f"  voice        {v.key}  ({v.label})   muted: {'yes' if self.tts.muted else 'no'}",
                 model_line("chat model", self.llm.model)]
        if self.llm.coder_model != self.llm.model:
            lines.append(model_line("coder model", self.llm.coder_model))
        if self.stt is not None:
            lines.append(f"  whisper      {self.cfg['stt']['model']} on {self.stt.device}")
        else:
            lines.append("  whisper      not loaded (text-only)")
        lines.append(f"  microphone   {self.mic.device_name if self.mic else 'none'}")
        lines.append(f"  workspace    {self.code.root}")
        lines.append(f"  idle timeout {int(self.idle_timeout)} s" if self.idle_timeout else "  idle timeout off")
        return "\n".join(lines) + "\n"

    def meta(self, cmd: str) -> bool:
        c = cmd.strip().lower()
        if c in ("/quit", "/exit", "/q"):
            self.running = False
        elif c == "/help":
            print(HELP)
        elif c == "/reset":
            self.llm.reset()
            log.info("Conversation history cleared.")
        elif c == "/mic":
            self.cmd_mic()
        elif c == "/status":
            print(self.status_text())
        else:
            return False
        return True

    # ---- main loop ---------------------------------------------------------------------
    def run(self) -> None:
        threading.Thread(target=self._stdin_loop, name="stdin", daemon=True).start()
        if self.mic and self.stt:
            threading.Thread(target=self._stt_loop, name="stt", daemon=True).start()

        log.info("== SLEEPING - say 'Jarvis' or type here. /help for commands ==")
        if not self.tts.muted:
            self.tts.say("Online.")
        while self.running:
            try:
                source, text, lang = self.events.get(timeout=0.5)
            except queue.Empty:
                self.check_inactivity()
                continue
            if text.startswith("/"):
                if self.meta(text):
                    continue
            try:
                self.handle(source, text, lang)
            except Exception:
                log.exception("Unexpected error while handling %r", text)
            finally:
                self.last_activity = time.time()

    def close(self) -> None:
        self.running = False
        self.tts.stop()
        if self.mic:
            self.mic.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Jarvis local voice assistant")
    ap.add_argument("--config", help="path to config.yaml")
    ap.add_argument("--text", action="store_true", help="typed input only (no microphone / Whisper)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    print(HELP)
    hardware.apply(cfg)                   # CPU fallback instead of a crash when there is no CUDA GPU
    jarvis = Jarvis(cfg, text_only=args.text)
    try:
        jarvis.setup()
    except LLMError as e:
        log.error("%s", e)
        return 1
    try:
        jarvis.run()
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Goodbye.")
        jarvis.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
