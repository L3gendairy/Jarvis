"""Ollama client: streaming chat with tool calling, model routing (chat vs coder), short-term history,
memory injection.

Standalone test:  python llm.py "What's the capital of Australia?"
                  python llm.py "Write a script that prints the first 20 Fibonacci numbers"   (routes to the coder)
"""
import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import ollama

import tools
from common import load_config, setup_logging

log = logging.getLogger("jarvis.llm")

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")

# ---- coding-request detection --------------------------------------------------------------
_CODE_NOUN = (r"script|program|programme|code|function|fonction|class|app|application|bot|game|jeu|tool|module|snippet|"
              r"algorithm|algorithme|scraper|calculator|calculatrice|website|webpage|cli|api|macro|automation")
_CODE_RE = re.compile(
    rf"""(
      \b(write|code|create|make|build|generate|develop|implement|program|script|put\ together)\b.{{0,45}}\b({_CODE_NOUN})\b
    | \bcode\s+(me|us|up|a|an|this|that|it)\b
    | \b(fix|debug|refactor|optimi[sz]e|rewrite|review)\b.{{0,35}}\b(code|script|program|bug|error|function|traceback|exception)\b
    | \b(run|execute|launch|test)\b.{{0,25}}\b(script|program|code|file|{_CODE_NOUN})\b
    | \b[\w\-]+\.py\b | \btraceback\b | \bpython\ (script|program|code|function|file)\b
    | \b(écris|ecris|code|programme|développe|developpe|crée|cree|fais|génère|genere)\b.{{0,45}}\b({_CODE_NOUN})\b
    | \b(corrige|répare|repare|débogue|debogue)\b.{{0,35}}\b(code|script|programme|bug|erreur|fonction)\b
    )""", re.I | re.X)
_FOLLOWUP_RE = re.compile(
    r"\b(it|that|this|script|file|code|program|run|execute|yes|yeah|yep|sure|go ahead|do it|oui|no|add|change|modify|"
    r"edit|update|fix|rename|again|faster|bigger|smaller|instead|also|show|retry|try|delete|install)\b", re.I)


_CALL_MARKER = re.compile(r'```|<tool_call>|\{\s*"name"')


def is_coding_request(text: str) -> bool:
    return bool(_CODE_RE.search(text))


def _parse_text_calls(text: str) -> list:
    """Some models (qwen2.5-coder) print tool calls as JSON text instead of using the tool-call channel.
    Recover them: {"name": ..., "arguments": {...}} possibly inside ```json fences or <tool_call> tags."""
    from types import SimpleNamespace
    known = set(tools.code.TOOL_NAMES) | {"web_search"}
    cleaned = re.sub(r"```(?:json)?|</?tool_call>", " ", text)
    decoder, calls, i = json.JSONDecoder(), [], 0
    while True:
        i = cleaned.find("{", i)
        if i < 0:
            break
        try:
            obj, end = decoder.raw_decode(cleaned, i)
        except json.JSONDecodeError:
            i += 1
            continue
        i = end
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name in known and isinstance(args, dict):
            calls.append(SimpleNamespace(function=SimpleNamespace(name=name, arguments=args)))
    return calls


class LLMError(RuntimeError):
    pass


def _find_ollama() -> str | None:
    """Path of the ollama executable: on PATH, or in the default per-user install folder (PATH may be stale)."""
    exe = shutil.which("ollama")
    if exe:
        return exe
    local = os.environ.get("LOCALAPPDATA")
    if local:
        default = Path(local) / "Programs" / "Ollama" / "ollama.exe"
        if default.is_file():
            return str(default)
    return None


def _tok(text: str) -> int:
    return len(text) // 4 + 4  # cheap estimate, good enough for trimming


def _forced_call_schema(names: list[str]) -> dict | None:
    """JSON schema (Ollama structured output) that only admits a call to one of the named tools."""
    options = []
    for entry in tools.code.SCHEMAS:
        fn = entry["function"]
        if fn["name"] in names:
            options.append({"type": "object", "required": ["name", "arguments"],
                            "properties": {"name": {"enum": [fn["name"]]}, "arguments": fn["parameters"]}})
    if not options:
        return None
    return options[0] if len(options) == 1 else {"anyOf": options}


def _brief(args: dict) -> str:
    """Compact tool-call arguments for the console (never dump whole source files)."""
    out = {}
    for k, v in (args or {}).items():
        out[k] = v if not isinstance(v, str) or len(v) <= 70 else v[:67].replace("\n", " ") + f"... [{len(v)} chars]"
    return json.dumps(out, ensure_ascii=False)


class LLM:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        l = cfg["llm"]
        self.model = l["model"]
        self.coder_model = l.get("coder_model") or self.model
        self.coder_ctx = int(l.get("coder_num_ctx", l.get("num_ctx", 8192)))
        self.coder_temp = float(l.get("coder_temperature", 0.2))
        self.coder_rounds = int(l.get("coder_max_tool_rounds", 14))
        self.followup_s = float(l.get("coding_followup_s", 180))
        self.coder_prompt = (l.get("coder_prompt") or "").strip()
        self.temperature = float(l.get("temperature", 0.7))
        self.num_ctx = int(l.get("num_ctx", 8192))
        self.keep_alive = l.get("keep_alive", "30m")
        self.max_rounds = int(l.get("max_tool_rounds", 3))
        self.budget = int(l.get("history_token_budget", 4500))
        self.persona = l["persona"].strip()
        self.extract_model = cfg.get("memory", {}).get("extract_model") or self.model
        self.client = ollama.Client(host=l.get("host", "http://127.0.0.1:11434"))
        self.history: list[dict] = []
        self._coding_until = 0.0
        self.last_model = self.model          # model used by the most recent chat() (for logs/tests)

    # ---- routing -----------------------------------------------------------------------
    def wants_coder(self, text: str) -> bool:
        """Coding request -> coder model. Short follow-ups right after a coding turn stay on the coder."""
        if is_coding_request(text):
            return True
        return time.time() < self._coding_until and len(text.split()) <= 14 and bool(_FOLLOWUP_RE.search(text))

    # ---- startup checks ----------------------------------------------------------------
    def ensure_server(self) -> None:
        """Verify Ollama is reachable (try to launch it if it is installed but not running)."""
        if self._ping():
            return
        exe = _find_ollama()
        if exe:
            log.info("Ollama is not running; starting 'ollama serve'...")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
            for _ in range(30):
                time.sleep(0.5)
                if self._ping():
                    return
        if not exe:
            raise LLMError("Ollama is not installed on this PC. Download it from https://ollama.com/download, "
                           "install it, then start Jarvis again.")
        raise LLMError("Ollama is installed but did not start. Launch the Ollama app (or run 'ollama serve' "
                       "in another terminal), then start Jarvis again.")

    def _ping(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def ensure_models(self) -> None:
        have = set()
        for m in self.client.list().models:
            name = getattr(m, "model", None) or getattr(m, "name", "")
            have.add(name)
            have.add(name.removesuffix(":latest"))
        needed = [self.model]
        for extra in (self.coder_model, self.extract_model):
            if extra not in needed:
                needed.append(extra)
        for name in needed:
            if name in have or name.removesuffix(":latest") in have:
                continue
            if not self.cfg["llm"].get("auto_pull", True):
                raise LLMError(f"Model '{name}' is not installed. Run:  ollama pull {name}")
            log.info("Model '%s' not found locally - pulling it now (this can take a while)...", name)
            last: dict[str, int] = {}                  # per layer: a pull downloads several, each from 0 to 100%
            try:
                for prog in self.client.pull(name, stream=True):
                    total, done = getattr(prog, "total", None), getattr(prog, "completed", None)
                    if total and done is not None:
                        layer = getattr(prog, "digest", "") or ""
                        pct = int(done * 100 / total)
                        if pct >= last.get(layer, -10) + 10:
                            log.info("  pulling %s: %d%%", name, pct)
                            last[layer] = pct
            except Exception as e:
                raise LLMError(f"Could not pull '{name}': {e}. Try:  ollama pull {name}") from e
            log.info("Model '%s' ready.", name)

    # ---- GPU placement (Ollama /api/ps) -------------------------------------------------
    def preload(self) -> None:
        """Load the chat model into memory now (same num_ctx as chat() so Ollama does not reload it later)."""
        self.client.generate(model=self.model, prompt="", keep_alive=self.keep_alive,
                             options={"num_ctx": self.num_ctx})

    def loaded_models(self) -> list[dict]:
        """Models currently loaded in Ollama: [{name, size, size_vram, gpu_pct}] (gpu_pct 0-100)."""
        out = []
        for m in self.client.ps().models:
            name = getattr(m, "model", None) or getattr(m, "name", "?")
            size = int(getattr(m, "size", 0) or 0)
            vram = int(getattr(m, "size_vram", 0) or 0)
            out.append({"name": name, "size": size, "size_vram": vram,
                        "gpu_pct": round(100 * vram / size) if size else 0})
        return out

    def gpu_report(self) -> list[dict]:
        """Log where each loaded model lives (GPU vs CPU) and warn if it is not 100% on the GPU."""
        models = self.loaded_models()
        if not models:
            log.warning("Ollama reports no loaded models (could not verify GPU placement).")
        for m in models:
            gb = m["size"] / 1e9
            log.info("VRAM check: %s  %.1f GB  GPU %d%% / CPU %d%%", m["name"], gb, m["gpu_pct"], 100 - m["gpu_pct"])
            if m["gpu_pct"] < 100:
                log.warning("WARNING: %s is only %d%% on the GPU - the rest runs on the CPU, so replies will be slow. "
                            "Free VRAM (close games/browsers) or switch to the 7B models in config.yaml: "
                            "model: qwen2.5:7b and coder_model: qwen2.5-coder:7b.", m["name"], m["gpu_pct"])
        return models

    # ---- prompt building ---------------------------------------------------------------
    def _system_prompt(self, facts: list[str], coding: bool) -> str:
        now = datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M")
        s = f"{self.persona}\n\nCurrent date and time: {now}."
        if coding and self.coder_prompt:
            s += "\n\n" + self.coder_prompt
        if facts:
            s += "\n\nWhat you remember about the user (use naturally, never recite unless asked):\n"
            s += "\n".join(f"- {f}" for f in facts)
        return s

    def _trim_history(self) -> None:
        def size(m):
            return _tok(m.get("content") or "") + sum(_tok(json.dumps(t)) for t in m.get("tool_calls") or [])
        while len(self.history) > 2 and sum(size(m) for m in self.history) > self.budget:
            self.history.pop(0)
            while self.history and self.history[0]["role"] != "user":  # never start mid-exchange
                self.history.pop(0)

    def reset(self) -> None:
        self.history.clear()

    # ---- chat --------------------------------------------------------------------------
    def chat(self, user_text: str, facts: list[str], on_sentence=lambda s: None, on_status=lambda s: None,
             should_stop=lambda: False, coding: bool = False, nudge_hook=lambda: None) -> str:
        """Run one user turn. Streams sentences to on_sentence as they complete; returns the full reply.
        coding=True routes to the coder model with the coding tools. If should_stop() turns true
        (user interrupted), generation is abandoned."""
        self.history.append({"role": "user", "content": user_text})
        self._trim_history()
        model = self.coder_model if coding else self.model
        num_ctx = self.coder_ctx if coding else self.num_ctx
        temperature = self.coder_temp if coding else self.temperature
        rounds = self.coder_rounds if coding else self.max_rounds
        system = {"role": "system", "content": self._system_prompt(facts, coding)}
        full_reply: list[str] = []
        tools_ok = True
        self.last_model = model
        nudges_left = 3          # times we may tell the model it stopped before finishing what was asked
        force_schema = None      # after a nudge, the next turn is constrained to a valid tool call

        for round_no in range(rounds + 1):
            messages = [system] + self.history
            fmt, force_schema = force_schema, None
            # If an unfinished instruction is pending (e.g. "...and run it"), hold this round's sentences back so
            # a premature "Want me to run it?" is never spoken when we are about to push the model to do it.
            buffered: list[str] = []
            defer = nudges_left > 0 and nudge_hook() is not None
            emit = buffered.append if defer else on_sentence
            use_tools = round_no < rounds and tools_ok and fmt is None
            try:
                stream = self.client.chat(
                    model=model, messages=messages, stream=True, format=fmt,
                    tools=tools.schemas(coding) if use_tools else None,
                    options={"temperature": temperature, "num_ctx": num_ctx},
                    keep_alive=self.keep_alive,
                )
                text, calls, pending = "", [], ""
                # Some models write tool calls as JSON text. From the first marker on (a ``` fence, <tool_call>, or
                # {"name"...), text is withheld from speech and parsed at the end instead.
                held, hold, started = "", False, False
                for chunk in stream:
                    if should_stop():
                        calls, pending, held, hold = [], "", "", False
                        break
                    msg = chunk.message
                    if msg.content:
                        text += msg.content
                        if hold:
                            held += msg.content
                            continue
                        pending += msg.content
                        marker = _CALL_MARKER.search(pending)
                        if not marker and not started and text.lstrip().startswith(("{", "[{")):
                            marker = re.search(r"\S", pending)
                        if marker:
                            before, held, pending, hold = pending[:marker.start()], pending[marker.start():], "", True
                            for sent in _SENTENCE_END.split(before):
                                if sent.strip():
                                    emit(sent.strip())
                            continue
                        parts = _SENTENCE_END.split(pending)
                        for sent in parts[:-1]:
                            if sent.strip():
                                emit(sent.strip())
                                started = True
                        pending = parts[-1]
                    if msg.tool_calls:
                        calls.extend(msg.tool_calls)
                if hold:
                    parsed = [] if calls else _parse_text_calls(held)
                    if parsed:
                        log.info("Recovered %d tool call(s) written as text by %s.", len(parsed), model)
                        calls = parsed
                        text = text[: len(text) - len(held)]          # the JSON is not part of the reply
                    else:
                        for sent in _SENTENCE_END.split(held):
                            if sent.strip():
                                emit(sent.strip())
                elif pending.strip():
                    emit(pending.strip())
            except ollama.ResponseError as e:
                if "does not support tools" in str(e.error).lower() and model != self.model:
                    log.warning("%s has no tool support; using %s for this request.", model, self.model)
                    model, tools_ok = self.model, False
                    continue
                raise LLMError(f"Ollama error: {e.error}") from e
            except Exception as e:
                raise LLMError(f"Could not reach Ollama ({e}). Is it still running?") from e

            if text.strip():
                full_reply.append(text.strip())
            assistant = {"role": "assistant", "content": text}
            if calls:
                assistant["tool_calls"] = [
                    {"function": {"name": c.function.name, "arguments": dict(c.function.arguments or {})}} for c in calls
                ]
            self.history.append(assistant)
            if not calls and not should_stop() and nudges_left > 0:
                nudge = nudge_hook()
                if nudge:
                    message, names = nudge if isinstance(nudge, tuple) else (nudge, [])
                    nudges_left -= 1
                    log.info("Model stopped early; nudging: %s", message)
                    log.info("  (discarded model reply: %r)", text.strip()[:300])
                    if text.strip():
                        full_reply.pop()
                    self.history.append({"role": "user", "content": message})
                    force_schema = _forced_call_schema(names) if names else None
                    continue                      # the held-back sentences are discarded
            for s in buffered:
                on_sentence(s)
            if not calls or should_stop():
                break
            for c in calls:
                args = dict(c.function.arguments or {})
                on_status(f"tool: {c.function.name}({_brief(args)})")
                result = tools.dispatch(c.function.name, args, self.cfg)
                self.history.append({"role": "tool", "content": result[:6000], "tool_name": c.function.name})

        if coding:
            self._coding_until = time.time() + self.followup_s
        return " ".join(full_reply).strip()

    def add_exchange_note(self, user_text: str, reply: str) -> None:
        """Record a handled-by-command exchange so the model keeps context."""
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        self._trim_history()

    def mark_coding(self) -> None:
        """A command (e.g. 'run snake.py') touched the workspace: keep follow-ups on the coder model."""
        self._coding_until = time.time() + self.followup_s

    def quick_json(self, prompt: str) -> str:
        """Single non-streaming JSON-mode call (fact extraction)."""
        try:
            r = self.client.chat(
                model=self.extract_model, messages=[{"role": "user", "content": prompt}], format="json",
                options={"temperature": 0.0, "num_ctx": 2048}, keep_alive=self.keep_alive,
            )
            return r.message.content or ""
        except Exception as e:
            log.debug("quick_json failed: %s", e)
            return ""


def _self_test() -> None:
    setup_logging("INFO")
    # routing (no Ollama needed)
    yes = ["write a script that prints primes", "Code me a snake game", "fix this code", "make a program to rename files",
           "write me a python script", "run snake.py", "écris un script pour trier des fichiers", "can you debug this traceback",
           "create a function that reverses a string"]
    no = ["what's the weather in Lyon", "tell me a joke", "who wrote the script for Alien", "how are you today",
          "remind me what code I like", "what's a good name for my dog"]
    for t in yes:
        assert is_coding_request(t), f"should route to coder: {t!r}"
    for t in no:
        assert not is_coding_request(t), f"should stay on chat model: {t!r}"
    log.info("Routing rules OK (%d coding / %d chat phrases).", len(yes), len(no))
    got = _parse_text_calls('```json\n{"name": "create_file", "arguments": {"path": "a.py", "content": "print(1)\\n"}}\n```')
    assert len(got) == 1 and got[0].function.name == "create_file" and got[0].function.arguments["path"] == "a.py"
    assert _parse_text_calls("Sure, here you go: {not json}") == []
    log.info("Text tool-call recovery OK.")

    cfg = load_config()
    llm = LLM(cfg)
    try:
        llm.ensure_server()
        llm.ensure_models()
    except LLMError as e:
        log.error("%s", e)
        sys.exit(1)
    q = " ".join(sys.argv[1:]) or "Who are you, in one sentence?"
    coding = llm.wants_coder(q)
    log.info("You: %s   [model: %s]", q, llm.coder_model if coding else llm.model)
    reply = llm.chat(q, [], on_sentence=lambda s: log.info("  [sentence] %s", s), on_status=lambda s: log.info("  %s", s),
                     coding=coding)
    log.info("Jarvis: %s", reply)
    sys.exit(0 if reply else 1)


if __name__ == "__main__":
    _self_test()
