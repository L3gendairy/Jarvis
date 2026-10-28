"""Shared helpers: config loading, logging, persistent settings, text utilities."""
import json
import logging
import re
import sys
import threading
import unicodedata
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("Missing Python packages (PyYAML). Start Jarvis with start.bat: it installs requirements.txt into .venv.")

ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def setup_logging(level: str = "INFO") -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "urllib3", "faster_whisper", "primp", "hpack",
                  "asyncio", "huggingface_hub", "filelock", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Settings:
    """Tiny JSON-backed key/value store (voice choice, mute state)."""

    def __init__(self, path: Path, defaults: dict):
        self.path = path
        self._lock = threading.Lock()
        self.data = dict(defaults)
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.data.update(json.load(f))
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError) as e:
            logging.getLogger("jarvis.settings").warning("Could not read %s (%s); using defaults.", path, e)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value) -> None:
        with self._lock:
            self.data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            tmp.replace(self.path)


_YES = {"yes", "yeah", "yep", "yup", "y", "sure", "ok", "okay", "go ahead", "do it", "run it", "please do", "proceed",
        "confirm", "affirmative", "of course", "absolutely", "go for it", "yes please", "yes run it", "yes do it",
        "oui", "ouais", "vas y", "d accord", "lance", "lance le", "oui vas y", "ok vas y"}
_NO = {"no", "nope", "nah", "n", "cancel", "stop", "don t", "do not", "dont", "never mind", "nevermind", "negative",
       "no thanks", "no thank you", "not now", "abort", "non", "annule", "pas maintenant", "non merci", "laisse tomber"}


def parse_yes_no(text: str) -> bool | None:
    """True for a yes, False for a no, None if the text is neither (EN + FR, voice or typed)."""
    n = normalize(text)
    if n in _YES:
        return True
    if n in _NO:
        return False
    words = n.split()
    if len(words) <= 4:                      # "yes sir", "no don't", "yes, go ahead please"
        if words and words[0] in ("yes", "yeah", "yep", "sure", "oui", "ok", "okay"):
            return True
        if words and words[0] in ("no", "nope", "nah", "non", "cancel", "stop"):
            return False
    return None


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation (keeps apostrophes/hyphens as spaces)."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text.replace("'", " ").replace("’", " "))
    return re.sub(r"\s+", " ", text).strip()
