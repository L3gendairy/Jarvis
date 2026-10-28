"""Wake-word and stop-phrase detection on Whisper transcripts (fuzzy, accent-insensitive).

Two entry points:
  find_for_wake(text)  - used while SLEEPING. Strict: the wake word must open the utterance ("Jarvis, ...",
                         "Hey Jarvis, ...") or the utterance must be very short (<= 3 words, e.g. "Jarvis",
                         "wake up Jarvis"). A "Jarvis" inside a long sentence does not wake him.
  strip_leading(text)  - used while ACTIVE: drops a leading "Jarvis," from a command.

Standalone test:  python wakeword.py
"""
import re
from difflib import SequenceMatcher

from common import normalize

# "Jarvis stop" and friends (English + French), compared after normalisation.
STOP_PHRASES = {
    "stop", "stop it", "stop listening", "that s all", "that is all", "go to sleep", "go back to sleep",
    "sleep", "never mind", "nevermind", "arrete", "arrete toi", "c est tout", "dors", "va dormir",
    "ca suffit", "stop ecoute", "arrete d ecouter",
}

# Words that may precede the wake word and still count as "at the start": "hey Jarvis", "ok Jarvis"...
_GREETINGS = {"hey", "hi", "hello", "ok", "okay", "yo", "oh", "so", "well", "salut", "dis", "dit", "ola", "bonjour", "eh"}
SHORT_UTTERANCE_WORDS = 3


class WakeWord:
    def __init__(self, cfg: dict):
        w = cfg["wake_word"]
        self.name = w.get("name", "jarvis").lower()
        self.aliases = {normalize(a) for a in w.get("aliases", [])} | {self.name}
        self.threshold = float(w.get("fuzzy_threshold", 0.78))

    def _is_wake(self, word: str) -> bool:
        if word in self.aliases:
            return True
        if len(word) < 4:
            return False
        return any(SequenceMatcher(None, word, a).ratio() >= self.threshold for a in self.aliases)

    @staticmethod
    def _words(text: str) -> list:
        return list(re.finditer(r"[\w'’-]+", text))

    @staticmethod
    def _tidy(rest: str) -> str:
        rest = re.sub(r"^[\s,.;:!?\-–—]+", "", rest)
        rest = re.sub(r"\s+([,.;:!?])", r"\1", rest)
        return rest.strip()

    def find(self, text: str) -> tuple[bool, str]:
        """Loose search: (heard_wake_word_anywhere, text_without_it). Not used for waking from sleep."""
        for m in self._words(text):
            if self._is_wake(normalize(m.group())):
                return True, self._tidy(text[: m.start()] + " " + text[m.end():])
        return False, text.strip()

    def find_for_wake(self, text: str) -> tuple[bool, str]:
        """Strict search used while sleeping. Returns (wake, command_after_the_wake_word)."""
        words = self._words(text)
        for i, m in enumerate(words):
            if not self._is_wake(normalize(m.group())):
                continue
            at_start = i == 0 or (i == 1 and normalize(words[0].group()) in _GREETINGS)
            if at_start:
                return True, self._tidy(text[m.end():])
            if len(words) <= SHORT_UTTERANCE_WORDS:
                return True, self._tidy(text[: m.start()] + " " + text[m.end():])
            return False, text.strip()           # first wake-like word is mid-sentence in a longer utterance
        return False, text.strip()

    def strip_leading(self, text: str) -> str:
        """Remove a leading 'Jarvis,' from text while already in active mode."""
        first = re.match(r"\s*[\w'’-]+", text)
        if first and self._is_wake(normalize(first.group())):
            return self._tidy(text[first.end():])
        return text.strip()

    @staticmethod
    def is_stop(text: str) -> bool:
        return normalize(text) in STOP_PHRASES


if __name__ == "__main__":
    ww = WakeWord({"wake_word": {"name": "jarvis", "aliases": ["jarvis", "jervis", "jarvice"]}})

    # --- sleeping: strict ---
    wake_cases = [
        ("Jarvis", True, ""),
        ("Jarvis.", True, ""),
        ("Jarvis, what time is it?", True, "what time is it?"),
        ("Hey Jarvis, what time is it?", True, "what time is it?"),
        ("Okay Jarvis stop", True, "stop"),
        ("Jervis allume la lumière", True, "allume la lumière"),
        ("wake up Jarvis", True, "wake up"),                    # short utterance
        ("what is the weather", False, "what is the weather"),
        ("I like service dogs", False, "I like service dogs"),
        ("I told Jarvis to turn the lights off", False, "I told Jarvis to turn the lights off"),
        ("so yesterday I was talking about Jarvis the movie", False, "so yesterday I was talking about Jarvis the movie"),
        ("what time is it Jarvis please", False, "what time is it Jarvis please"),
        ("my friend Travis said hello", False, "my friend Travis said hello"),    # no longer an alias
        ("Gervais is a comedian", False, "Gervais is a comedian"),                # no longer an alias
    ]
    for text, exp_wake, exp_rest in wake_cases:
        wake, rest = ww.find_for_wake(text)
        assert wake == exp_wake and rest == exp_rest, (text, wake, rest)

    # --- loose find / active mode ---
    assert ww.find("what time is it Jarvis please")[0] is True
    assert ww.is_stop("Stop.") and ww.is_stop("Arrête !") and not ww.is_stop("stop the music please")
    assert ww.strip_leading("Jarvis, tell me a joke") == "tell me a joke"
    assert ww.strip_leading("tell me a joke Jarvis") == "tell me a joke Jarvis"
    assert ww.strip_leading("Jarvis") == ""
    print(f"OK: wakeword tests passed ({len(wake_cases)} wake cases)")
