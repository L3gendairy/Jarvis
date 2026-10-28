"""Voice/text command router (English + French). Handles control commands without the LLM.

Standalone test:  python commands.py
"""
import logging
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from common import normalize
from tools.code import CodeError
from wakeword import WakeWord

log = logging.getLogger("jarvis.commands")


@dataclass
class Result:
    reply: str                  # spoken (and printed)
    action: str | None = None   # "sleep" | "quit" | None
    display: str = ""           # extra text printed to the console only (e.g. numbered voice list)
    llm_prompt: str = ""        # if set, main.py sends this to the (coder) LLM after speaking the reply


_VOICE = r"\bvoi(?:ce|x)s?\b"
_LIST_RE = re.compile(r"\b(list|show|available|which voices|what voices|quelles voix|liste|enumere)\b")
_CURRENT_RE = re.compile(r"\b(what|which|quelle)\b.*" + _VOICE + r".*\b(using|currently|right now|now|actuellement|utilises|have)\b")
_CHANGE_RE = re.compile(r"\b(change|switch|swap|next|another|different|use|set|activate|put on|passe|mets|utilise|prends|"
                        r"talk like|speak like|sound like|go to|select|choose)\b")
_UNMUTE_RE = re.compile(r"^(please )?(un ?mute( yourself)?|speak (to me )?again|talk (to me )?again|you can (talk|speak)( again)?|"
                        r"reactive (le son|la voix)|tu peux parler|remets (le son|la voix))$")
_MUTE_RE = re.compile(r"^(please )?(mute( yourself)?|be (quiet|silent)|text only|silence|tais toi|mets toi en muet|coupe le son|coupe la voix)$")
_QUIT_RE = re.compile(r"^(shut ?down|shutdown|exit|quit|power down|goodbye|eteins toi|ferme toi|au revoir)$")
_KNOW_RE = re.compile(r"(what (do|did|have) you (know|remember|learned|learnt)|what do you know|tell me what you know|"
                      r"ce que tu sais|que sais tu|qu est ce que tu sais)")
_REMEMBER_RE = re.compile(
    r"^\W*(?:please\s+)?(?:remember|memorize|memorise|note|keep in mind|retiens|souviens[- ]toi|rappelle[- ]toi|n['’]oublie pas)"
    r"(?:\s+(?:that|this|de|que|qu['’]))?\s*(?P<fact>.+?)\W*$", re.I | re.S)
_FORGET_RE = re.compile(
    r"^\W*(?:please\s+)?(?:forget|delete|erase|oublie|efface)(?:\s+(?:about|that|what i said about|que|à propos de|a propos de))?\s+(?P<what>.+?)\W*$",
    re.I | re.S)
_WIPE_RE = re.compile(r"^(everything|all|everything about me|all of it|tout|tout sur moi)$")
_CONFIRM_RE = re.compile(r"^(confirm|confirmed|yes|yes do it|do it|go ahead|oui|vas y|confirme)$")

_ACKS = ["Noted.", "Consider it filed.", "Got it. Stored.", "Committed to memory."]

# ---- workspace commands: open workspace / list files / run [file] / show [file] ----
_OPEN_WS_RE = re.compile(r"^\W*(?:please\s+)?(?:open|show|launch)(?:\s+me)?(?:\s+(?:the|my))?\s+work\s?space(?:\s+folder)?\W*$", re.I)
_LIST_FILES_RE = re.compile(r"^\W*(?:please\s+)?(?:list|show)(?:\s+me)?(?:\s+(?:the|my|all))*(?:\s+(?:workspace|work space))?\s+files\W*$"
                            r"|^\W*what(?:'s|\s+is)?\s+in\s+(?:the|my)\s+work\s?space\W*$", re.I)
_RUN_RE = re.compile(r"^\W*(?:please\s+)?(?:run|execute|launch)(?:\s+the)?(?:\s+(?:file|script|program))?\s+(?P<f>[\w .\\/\-]+?)\W*$", re.I)
_SHOW_RE = re.compile(r"^\W*(?:please\s+)?(?:show|display|print|read|cat)(?:\s+me)?(?:\s+the)?(?:\s+(?:file|script|code|program|source))?\s+(?P<f>[\w .\\/\-]+?)\W*$", re.I)
_DOT_EXT_RE = re.compile(r"\s+dot\s+(py|txt|md|json|csv|html|js)\b", re.I)

_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six ": 6}
_NUM_SPOKEN = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
# Words that carry no voice name once a voice command is recognised.
_FILLER = set("""use the a an to switch change set activate put on pass passe mets met utilise prends prend talk speak
sound like as your my you please now again another different next other new voice voices voix de la le les l can could
would will it that me for in with parle parler sois be go select choose number numero a au un une and then""".split())


def _phon(word: str) -> str:
    """Crude phonetic key so Whisper's 'sin'/'sinn'/'cyn' all compare equal."""
    w = re.sub(r"c(?=[eiy])", "s", word)
    w = w.replace("ph", "f").replace("y", "i").replace("kn", "n")
    return re.sub(r"(.)\1+", r"\1", w)


def _third_person(fact: str) -> str:
    """'my dog is called Max' -> "The user's dog is called Max" (English only; others stay as said)."""
    f = fact.strip()
    subs = [
        (r"\bI am\b", "the user is"), (r"\bI['’]m\b", "the user is"), (r"\bI['’]ve\b", "the user has"),
        (r"\bI['’]ll\b", "the user will"), (r"\bI['’]d\b", "the user would"), (r"\bmy\b", "the user's"),
        (r"\bmine\b", "the user's"), (r"\bmyself\b", "the user"), (r"\bme\b", "the user"), (r"\bI\b", "the user"),
    ]
    out = f
    for pat, rep in subs:
        out = re.sub(pat, rep, out, flags=re.I)
    if out != f:
        out = out[0].upper() + out[1:]
    else:
        out = f"The user asked me to remember: {f}"
    return out if out.endswith((".", "!", "?")) else out + "."


def _second_person(fact: str) -> str:
    out = re.sub(r"\bthe user['’]s\b", "your", fact, flags=re.I)
    out = re.sub(r"\bthe user is\b", "you are", out, flags=re.I)
    out = re.sub(r"\bthe user has\b", "you have", out, flags=re.I)
    out = re.sub(r"\bthe user\b", "you", out, flags=re.I)
    return out[0].upper() + out[1:] if out else out


class Commands:
    def __init__(self, tts, memory, wake: WakeWord, code=None, on_code_used=lambda: None):
        self.tts = tts
        self.memory = memory
        self.wake = wake
        self.code = code                      # tools.code.CodeTools (workspace) or None
        self.on_code_used = on_code_used      # lets the LLM router know the user is in a coding session
        self._pending_wipe = False

    # ---- workspace commands ------------------------------------------------------------
    def _workspace_command(self, raw: str) -> Result | None:
        if self.code is None:
            return None
        raw = _DOT_EXT_RE.sub(r".\1", raw)
        if _OPEN_WS_RE.match(raw):
            import os
            try:
                os.startfile(str(self.code.root))
            except (OSError, AttributeError) as e:
                return Result(f"I couldn't open the folder: {e}. It's at {self.code.root}.")
            log.info("Opened workspace: %s", self.code.root)
            return Result("Opening the workspace.")
        if _LIST_FILES_RE.match(raw):
            files = self.code._all_files()
            if not files:
                return Result("The workspace is empty.")
            shown = ", ".join(files[:8]) + (f", and {len(files) - 8} more" if len(files) > 8 else "")
            return Result(f"{len(files)} file{'s' if len(files) != 1 else ''} in the workspace: {shown}.",
                          display="Workspace files:\n" + self.code.list_files())
        m = _RUN_RE.match(raw)
        if m:
            rel = self.code.resolve_name(m.group("f"))
            if rel is None:
                return None                   # "run the numbers" etc. -> normal chat
            self.on_code_used()
            try:
                res = self.code.run_file(rel)
            except CodeError as e:
                return Result(f"I couldn't run it: {e}")
            if res.status == "declined":
                return Result("Cancelled. Nothing was run.")
            if res.status == "ok":
                lines = [l for l in res.stdout.strip().splitlines() if l.strip()]
                if not lines:
                    return Result(f"{rel} ran cleanly, with no output.")
                if len(lines) <= 3 and sum(len(l) for l in lines) <= 160:
                    return Result(f"{rel} ran cleanly. It printed: " + " ".join(lines))
                return Result(f"{rel} ran cleanly and printed {len(lines)} lines. See the console.")
            if res.status == "timeout":
                return Result(f"{rel} was still running after {self.code.timeout} seconds, so I stopped it.")
            err = (res.stderr.strip().splitlines() or ["unknown error"])[-1]
            try:
                source = self.code.read_file(rel)
            except CodeError:
                source = ""
            return Result(f"{rel} failed: {err[:140]}. Let me take a look.",
                          llm_prompt=(f"I ran {rel} from the workspace and it failed with exit code {res.exit_code}.\n"
                                      f"stderr:\n{res.stderr[-1500:]}\n\nCurrent contents of {rel}:\n{source}\n\n"
                                      f"Fix the bug in the file with edit_file, then run it again."))
        m = _SHOW_RE.match(raw)
        if m:
            rel = self.code.resolve_name(m.group("f"))
            if rel is None:
                return None
            self.on_code_used()
            self.code.ui.show_code(rel, self.code.read_file(rel))
            return Result(f"Showing {rel} in the console.")
        return None

    # ---- voice helpers -----------------------------------------------------------------
    def _keys(self) -> list[str]:
        return list(self.tts.voices)

    def _name(self, key: str) -> str:
        return key.title()

    def _voice_state(self) -> str:
        v = self.tts.voices[self.tts.voice_key]
        return f"{v.key} (model={getattr(v, 'model', '?')}, effects={len(getattr(v, 'effects', []) or [])})"

    def _numbered_list(self) -> tuple[str, str]:
        """Return (console text, spoken text) for the numbered voice list."""
        keys = self._keys()
        lines = []
        for i, k in enumerate(keys, 1):
            v = self.tts.voices[k]
            mark = "   <- active" if k == self.tts.voice_key else ""
            lines.append(f"  {i}. {k:<8} {getattr(v, 'label', k)}{mark}")
        spoken = " ".join(f"{_NUM_SPOKEN.get(i, i)}, {self._name(k)}." for i, k in enumerate(keys, 1))
        return "Available voices:\n" + "\n".join(lines), spoken

    def _match_voice(self, tokens: list[str]) -> str | None:
        """Fuzzy/phonetic match of spoken tokens against voice keys and aliases."""
        best, best_score = None, 0.0
        for tok in tokens:
            if len(tok) < 3:
                continue
            pt = _phon(tok)
            for key in self._keys():
                names = [key] + list(getattr(self.tts.voices[key], "aliases", []) or [])
                for name in names:
                    pn = _phon(normalize(name))
                    score = 1.0 if (tok == normalize(name) or pt == pn) else SequenceMatcher(None, pt, pn).ratio()
                    if score > best_score:
                        best, best_score = key, score
        return best if best_score >= 0.75 else None

    def _switch_to(self, key: str, heard: str, how: str) -> Result:
        try:
            self.tts.set_voice(key)
        except Exception as e:
            log.error("Voice command %r -> %s FAILED to load '%s': %s | active voice: %s",
                      heard, how, key, e, self._voice_state())
            return Result(f"I couldn't load the {self._name(key)} voice. I'm staying with {self._name(self.tts.voice_key)}.")
        log.info("Voice command %r -> %s | active voice now: %s", heard, how, self._voice_state())
        return Result(f"Voice changed to {self._name(key)}.")

    def _voice_command(self, n: str) -> Result | None:
        """Handle list / current / change / select-by-name / select-by-number. None = not a voice command."""
        words = n.split()
        has_voice_word = bool(re.search(_VOICE, n))
        number = None
        m = (re.search(r"\b(?:voice|voix|number|numero)\s+(\d+|" + "|".join(k for k in _NUM_WORDS if " " not in k) + r")\b", n)
             or re.search(r"\b(\d+)(?:st|nd|rd|th|e|eme)?\s+voi(?:ce|x)\b", n))
        if m:
            g = m.group(1)
            number = int(g) if g.isdigit() else _NUM_WORDS.get(g)

        if not has_voice_word:
            return None
        if len(words) > 10:                                    # a sentence, not a command
            return None

        if _LIST_RE.search(n):
            console, spoken = self._numbered_list()
            log.info("Voice command %r -> list | active voice: %s", n, self._voice_state())
            return Result(f"I have {len(self._keys())} voices. {spoken} I'm using {self._name(self.tts.voice_key)}.", display=console)
        if _CURRENT_RE.search(n):
            log.info("Voice command %r -> current | active voice: %s", n, self._voice_state())
            return Result(f"I'm using the {self._name(self.tts.voice_key)} voice.")

        has_verb = bool(_CHANGE_RE.search(n))
        if number is not None and (has_verb or len(words) <= 4):
            keys = self._keys()
            if 1 <= number <= len(keys):
                return self._switch_to(keys[number - 1], n, f"select #{number}")
            console, _ = self._numbered_list()
            log.info("Voice command %r -> bad number %s | active voice: %s", n, number, self._voice_state())
            return Result(f"I only have {len(keys)} voices.", display=console)

        leftover = [w for w in words if w not in _FILLER and not w.isdigit() and w not in _NUM_WORDS]
        if not leftover:
            if has_verb:
                keys = self._keys()
                nxt = keys[(keys.index(self.tts.voice_key) + 1) % len(keys)]
                return self._switch_to(nxt, n, "cycle")
            return None
        key = self._match_voice(leftover)
        if key is not None:
            return self._switch_to(key, n, f"select '{key}'")
        if has_verb:
            console, spoken = self._numbered_list()
            log.info("Voice command %r -> unknown voice %r | active voice: %s", n, " ".join(leftover), self._voice_state())
            return Result(f"I don't have a voice called {' '.join(leftover)}. {spoken}", display=console)
        return None                                            # e.g. "the voice of reason" -> normal chat

    # ---- router ------------------------------------------------------------------------
    def handle(self, text: str) -> Result | None:
        """Return a Result if `text` was a command, else None (-> send to the LLM)."""
        raw = text.strip()
        n = normalize(raw)
        if not n:
            return Result("")

        if self._pending_wipe:
            self._pending_wipe = False
            if _CONFIRM_RE.match(n):
                count = self.memory.forget_all()
                return Result(f"Done. {count} memories erased. A fresh start.")
            return Result("Wipe cancelled. Your secrets are safe with me.")

        if self.wake.is_stop(raw):
            return Result("Standing by.", "sleep")
        if _QUIT_RE.match(n):
            return Result("Shutting down. Until next time.", "quit")

        if _UNMUTE_RE.match(n):
            self.tts.set_muted(False)
            return Result("Voice back online.")
        if _MUTE_RE.match(n):
            self.tts.set_muted(True)
            return Result("Muted. Text only from here.")

        res = self._workspace_command(raw)
        if res is not None:
            return res

        res = self._voice_command(n)
        if res is not None:
            return res

        if _KNOW_RE.search(n) and re.search(r"\b(me|moi|know|sais|remember)\b", n) and len(n.split()) <= 9:
            facts = self.memory.all()
            if not facts:
                return Result("Nothing yet. A blank slate. Tell me something about yourself.")
            shown = facts[-8:]
            more = f" Plus {len(facts) - len(shown)} older ones." if len(facts) > len(shown) else ""
            return Result(f"I know {len(facts)} thing{'s' if len(facts) != 1 else ''}. "
                          + " ".join(_second_person(f) for f in shown) + more)

        m = _REMEMBER_RE.match(raw)
        if m and len(m.group("fact").split()) >= 2:
            fact = _third_person(m.group("fact"))
            _, created = self.memory.add(fact, source="user")
            return Result(random.choice(_ACKS) if created else "I already had that one.")

        m = _FORGET_RE.match(raw)
        if m:
            what = normalize(m.group("what"))
            if _WIPE_RE.match(what):
                self._pending_wipe = True
                return Result(f"That erases all {self.memory.count()} memories. Say confirm to proceed.")
            removed = self.memory.forget(m.group("what"))
            if not removed:
                return Result("I couldn't find anything matching that.")
            return Result(f"Forgotten. {len(removed)} item{'s' if len(removed) > 1 else ''} removed.")

        return None


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    from memory import Memory

    class FakeVoice:
        def __init__(self, key, aliases=()):
            self.key, self.label, self.aliases, self.model, self.effects = key, key, list(aliases), "m", []

    class FakeTTS:
        def __init__(self):
            self.voices = {"jarvis": FakeVoice("jarvis"), "ryan": FakeVoice("ryan"), "amy": FakeVoice("amy"),
                           "cyn": FakeVoice("cyn", ["sin", "sinn", "sync"])}
            self.voice_key, self.muted = "jarvis", False
        def set_muted(self, m): self.muted = m
        def set_voice(self, k): self.voice_key = k

    with tempfile.TemporaryDirectory() as d:
        mem = Memory(Path(d) / "m.db")
        tts = FakeTTS()
        c = Commands(tts, mem, WakeWord({"wake_word": {}}))
        # cycling
        assert c.handle("change voice").reply == "Voice changed to Ryan." and tts.voice_key == "ryan"
        assert c.handle("Change voice.").reply == "Voice changed to Amy."
        # direct selection with Whisper variations
        for phrase in ["use the cyn voice", "Use cyn voice", "switch to the Cyn voice.", "sin voice", "use the sinn voice",
                       "mets la voix cyn", "voice 4", "Voice four", "use the 4th voice"]:
            tts.voice_key = "jarvis"
            assert c.handle(phrase).reply == "Voice changed to Cyn.", phrase
            assert tts.voice_key == "cyn", phrase
        tts.voice_key = "jarvis"
        assert c.handle("voice 2").reply == "Voice changed to Ryan."
        assert c.handle("use the ryan voice").reply == "Voice changed to Ryan."
        # unknown name / number
        r = c.handle("use the foo voice")
        assert r.reply.startswith("I don't have a voice called foo") and "1. jarvis" in r.display and tts.voice_key == "ryan"
        assert c.handle("voice 9").reply == "I only have 4 voices."
        # listing
        r = c.handle("list voices")
        assert "One, Jarvis. Two, Ryan. Three, Amy. Four, Cyn." in r.reply and "4. cyn" in r.display
        assert "Ryan voice" in c.handle("what voice are you using right now").reply
        # not voice commands
        assert c.handle("tell me what the voice of reason would say") is None
        assert c.handle("my voice is tired today") is None
        # other commands
        assert c.handle("mute").reply.startswith("Muted") and tts.muted
        assert c.handle("unmute").reply.startswith("Voice back") and not tts.muted
        assert c.handle("stop").action == "sleep"
        assert c.handle("remember that my dog is called Max") and mem.all() == ["The user's dog is called Max."]
        assert c.handle("retiens que j'habite à Lyon") and mem.count() == 2
        assert "dog is called Max" in c.handle("what do you know about me?").reply
        assert c.handle("forget my dog").reply.startswith("Forgotten") and mem.count() == 1
        assert "erases all 1" in c.handle("forget everything").reply
        assert "erased" in c.handle("confirm").reply and mem.count() == 0
        assert c.handle("what's the weather in Paris?") is None

        # workspace commands
        from tools.code import CodeTools

        class UI:
            answers = [False, True]
            def confirm(self, q): return self.answers.pop(0)
            def show_code(self, rel, code): self.shown = rel
            def show_warning(self, rel, findings): pass
            def show_output(self, r): pass
        ui = UI()
        code = CodeTools({"code": {"workspace": str(Path(d) / "ws")}}, ui)
        code.create_file("fib.py", "print('a b c')\n")
        cw = Commands(tts, mem, WakeWord({"wake_word": {}}), code)
        assert "1 file in the workspace: fib.py" in cw.handle("list files").reply
        assert cw.handle("show fib dot py").reply == "Showing fib.py in the console." and ui.shown == "fib.py"
        assert cw.handle("run fib.py").reply == "Cancelled. Nothing was run."
        assert "ran cleanly. It printed: a b c" in cw.handle("Run fib").reply
        assert cw.handle("run the numbers please") is None and cw.handle("show me the weather") is None
        code.create_file("bad.py", "raise SystemExit(3)\n")
        ui.answers = [True]
        r = cw.handle("run bad.py")
        assert r.llm_prompt and "bad.py" in r.llm_prompt and "failed" in r.reply
        mem.close()
    print("OK: commands tests passed")
