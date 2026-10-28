"""Coding tools: a sandboxed workspace folder where Jarvis can create, read, edit, list and delete files,
run Python scripts (60 s timeout) and install pip packages.

Safety rules enforced here (not left to the LLM):
  * every path is resolved and must stay inside the workspace (no '..', no absolute paths elsewhere, no drive tricks)
  * only text/source extensions can be written; executables and scripts for other shells are refused
  * running a script, deleting a file and installing a package ALL require explicit user confirmation
  * the full source is printed to the console before a script is run
  * at most `max_run_failures` failed runs per request

Standalone test:  python -m tools.code
"""
import ast
import logging
import os
import re
import shlex
import venv as _venv
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from common import resolve

log = logging.getLogger("jarvis.code")

TOOL_NAMES = {"create_file", "read_file", "edit_file", "list_files", "delete_file", "run_python", "pip_install"}

ALLOWED_EXT = {".py", ".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".html", ".css", ".js", ".ts", ".toml",
               ".ini", ".cfg", ".xml", ".sql", ".log", ".tsv"}
_PKG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*(\[[A-Za-z0-9_,\-]+\])?([=<>~!]=?[A-Za-z0-9.*]+(,[=<>~!]=?[A-Za-z0-9.*]+)*)?$")


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


_PATH = {"type": "string", "description": "File path relative to the workspace, e.g. 'snake.py' or 'utils/helpers.py'."}

SCHEMAS = [
    _fn("create_file", "Create a file in the workspace (overwrites if it exists) with the COMPLETE source code. "
        "The code goes here, never in your spoken reply.",
        {"path": _PATH, "content": {"type": "string", "description": "Full file contents."}}, ["path", "content"]),
    _fn("read_file", "Read a file from the workspace.", {"path": _PATH}, ["path"]),
    _fn("edit_file", "Edit a file by replacing one exact occurrence of old_text with new_text. "
        "old_text must match the file exactly and be unique; read_file first if unsure.",
        {"path": _PATH, "old_text": {"type": "string", "description": "Exact existing text to replace."},
         "new_text": {"type": "string", "description": "Replacement text."}}, ["path", "old_text", "new_text"]),
    _fn("list_files", "List all files in the workspace.", {}, []),
    _fn("delete_file", "Delete a file from the workspace (asks the user to confirm).", {"path": _PATH}, ["path"]),
    _fn("run_python", "Run a Python script from the workspace and return its exit code, stdout and stderr "
        "(60 s timeout; no keyboard input available). The user is asked to confirm and sees the code first. "
        "If it fails, read the error, fix the file with edit_file or create_file, and run again.",
        {"path": _PATH, "args": {"type": "string", "description": "Optional command-line arguments."}}, ["path"]),
    _fn("pip_install", "Install a Python package with pip (asks the user to confirm). Prefer the standard library.",
        {"package": {"type": "string", "description": "Package name, e.g. 'requests' or 'pygame==2.5.2'."}}, ["package"]),
]


_RUN_REQUEST_RE = re.compile(
    r"\b(run|execute|launch|test)\s+(it|them|that|this|again|the\s+(script|program|code|file))\b"
    r"|\b(and|then|puis|et)\s+(run|test|execute|lance|exécute|execute)\b|\brun\s+[\w\-]+\.py\b", re.I)


_WRITE_REQUEST_RE = re.compile(
    r"\b(write|create|make|build|generate|code|develop|implement|écris|ecris|crée|cree|fais|génère|genere)\b"
    r".{0,45}\b(script|program|programme|code|function|fonction|class|app|application|bot|game|jeu|tool|module|"
    r"snippet|file|scraper|calculator|calculatrice)\b", re.I)


_EDIT_REQUEST_RE = re.compile(r"\b(fix|debug|repair|correct|edit|change|modify|update|refactor|corrige|répare|repare)\b", re.I)


class CodeError(Exception):
    """A refused or failed operation; the message is returned to the LLM verbatim."""


@dataclass
class RunResult:
    status: str                     # ok | failed | timeout | declined | refused
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    message: str = ""

    def for_llm(self) -> str:
        if self.status == "declined":
            return "The user declined to run the script. Do not run it again unless they ask."
        if self.status == "refused":
            return self.message
        head = {"ok": "Script finished successfully.", "failed": "Script FAILED.",
                "timeout": self.message}[self.status]
        parts = [head, f"Exit code: {self.exit_code}"]
        if self.stdout.strip():
            parts.append("--- stdout ---\n" + self.stdout[:3000])
        if self.stderr.strip():
            parts.append("--- stderr ---\n" + self.stderr[-3000:])
        if self.message and self.status == "failed":
            parts.append(self.message)
        return "\n".join(parts)


RISKY_QUESTION = "This script does something risky, check the console. Run it?"

_NET_MODULES = {"socket", "requests", "urllib", "urllib3", "http", "httpx", "aiohttp", "ftplib", "smtplib", "telnetlib",
                "websocket", "websockets", "paramiko", "xmlrpc", "poplib", "imaplib", "asyncio.streams"}
_DELETE_CALLS = {"os.remove", "os.unlink", "os.rmdir", "os.removedirs", "shutil.rmtree", "send2trash.send2trash"}
_DELETE_ATTRS = {"unlink", "rmdir"}                               # Path(...).unlink() etc.
_MOVE_CALLS = {"os.rename", "os.replace", "shutil.move"}
_PROC_CALLS = {"os.system", "os.popen", "os.startfile", "subprocess.run", "subprocess.call", "subprocess.Popen",
               "subprocess.check_call", "subprocess.check_output", "os.execv", "os.execl", "os.execvp", "os.spawnl",
               "os.spawnv"}
_DYNAMIC_CALLS = {"eval", "exec", "__import__"}


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _outside_literal(path: str) -> bool:
    p = path.strip()
    return (os.path.isabs(p) or p.startswith(("~", "\\\\", "/")) or ".." in re.split(r"[\\/]", p)
            or bool(re.match(r"^[A-Za-z]:", p)) or bool(re.search(r"%\w+%", p)))


def scan_risks(code: str) -> list[str]:
    """Best-effort static scan of a Python script. Returns human-readable findings ('line N: ...').
    It cannot see code in imported modules or built at run time, so it is a warning aid, not a sandbox."""
    findings: list[str] = []

    def add(node, msg):
        findings.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is None:                                     # unparsable: fall back to plain text patterns
        for pat, msg in [(r"\b(os\.remove|os\.unlink|shutil\.rmtree|os\.rmdir)\b", "deletes files"),
                         (r"\b(subprocess|os\.system|os\.popen)\b", "runs other programs"),
                         (r"\b(socket|requests|urllib|http\.client|httpx)\b", "uses the network"),
                         (r"\bwinreg\b|HKEY_", "touches the Windows registry")]:
            if re.search(pat, code):
                findings.append(f"(file does not parse; text match) {msg}")
        return findings

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for mod in mods:
                root = mod.split(".")[0]
                if root in _NET_MODULES:
                    add(node, f"uses the network (imports {mod})")
                elif root in ("winreg", "_winreg"):
                    add(node, "accesses the Windows registry (winreg)")
                elif root == "subprocess":
                    add(node, "runs other programs (imports subprocess)")
                elif root == "ctypes":
                    add(node, "calls native code (ctypes)")
        elif isinstance(node, ast.Call):
            name = _dotted(node.func)
            attr = node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in _DELETE_CALLS or attr in _DELETE_ATTRS:
                add(node, f"deletes files ({name or attr})")
            elif name in _MOVE_CALLS:
                add(node, f"moves/renames files ({name})")
            elif name in _PROC_CALLS or name.startswith(("os.exec", "os.spawn")):
                add(node, f"runs other programs ({name})")
            elif name in _DYNAMIC_CALLS:
                add(node, f"executes dynamically built code ({name})")
            elif name in ("open", "io.open") or attr == "open":
                _scan_open(node, name or attr, add)
            elif attr in ("write_text", "write_bytes") and isinstance(node.func.value, ast.Call):
                _scan_path_write(node, add)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "HKEY_" in node.value or re.search(r"\breg\s+(add|delete|import)\b", node.value, re.I):
                add(node, "accesses the Windows registry")
    # de-duplicate, keep order
    seen, out = set(), []
    for f in findings:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _scan_open(node: ast.Call, name: str, add) -> None:
    args, kw = node.args, {k.arg: k.value for k in node.keywords}
    if isinstance(node.func, ast.Attribute) and name != "io.open":
        # Path(x).open(mode): the path is the receiver's first argument, the mode our first argument
        recv = node.func.value
        path_node = recv.args[0] if isinstance(recv, ast.Call) and recv.args else None
        mode_node = args[0] if args else kw.get("mode")
    else:
        path_node = args[0] if args else kw.get("file")
        mode_node = args[1] if len(args) > 1 else kw.get("mode")
    mode =mode_node.value if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str) else "r"
    writing = any(c in mode for c in "wax+")
    if isinstance(path_node, ast.Constant) and isinstance(path_node.value, str):
        if _outside_literal(path_node.value):
            add(node, f"{'writes' if writing else 'reads'} a file outside the workspace ({path_node.value!r})")
    elif path_node is not None and writing:
        add(node, "writes to a computed path (cannot verify it stays inside the workspace)")


def _scan_path_write(node: ast.Call, add) -> None:
    recv = node.func.value
    if recv.args and isinstance(recv.args[0], ast.Constant) and isinstance(recv.args[0].value, str) \
            and _outside_literal(recv.args[0].value):
        add(node, f"writes a file outside the workspace ({recv.args[0].value!r})")


class ConsoleUI:
    """Minimal fallback UI (used by self-tests); main.py supplies the real one."""

    def confirm(self, question: str) -> bool:
        return input(f"{question} [y/n] ").strip().lower().startswith("y")

    def show_code(self, rel: str, code: str) -> None:
        print(f"\n---- {rel} ----\n{code}\n----")

    def show_warning(self, rel: str, findings: list[str]) -> None:
        print(f"\nWARNING: {rel} does risky things:")
        for f in findings:
            print(f"  - {f}")

    def show_output(self, result: RunResult) -> None:
        print(result.for_llm())


class CodeTools:
    def __init__(self, cfg: dict, ui=None):
        c = cfg.get("code", {})
        self.root = resolve(c.get("workspace", "workspace"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()
        self.timeout = int(c.get("run_timeout_s", 60))
        self.max_failures = int(c.get("max_run_failures", 3))
        self.max_bytes = int(c.get("max_file_kb", 200)) * 1024
        self.ui = ui or ConsoleUI()
        self.venv_dir = self.root / ".venv"        # the workspace's own environment (never Jarvis's .venv)
        self.run_failures = 0
        self.want_run = False        # the current request asks for the code to be run
        self.ran = False
        self.last_written = ""       # last .py file created/edited this request
        self.want_write = False      # the current request asks for new code to be written
        self.want_edit = False       # ... or for existing code to be fixed/changed
        self.edited = False          # a file was written this request
        self.tool_calls = 0

    def reset_turn(self, user_text: str = "") -> None:
        """Call at the start of each user request: the 3-attempt budget is per request."""
        self.run_failures = 0
        self.ran = False
        self.last_written = ""
        self.tool_calls = 0
        self.edited = False
        self.want_run = bool(_RUN_REQUEST_RE.search(user_text or ""))
        self.want_write = bool(_WRITE_REQUEST_RE.search(user_text or ""))
        self.want_edit = bool(_EDIT_REQUEST_RE.search(user_text or ""))

    def nudge(self) -> tuple[str, list[str]] | None:
        """(message, tools the next turn is constrained to) when the model stops before doing what was asked:
        it only talked, or it saved the code but never ran it although the user asked for that."""
        if self.want_write and self.tool_calls == 0:
            return ("You have not called any tool, so NOTHING has been saved. Call create_file now with the complete "
                    "code.", ["create_file"])
        if self.want_edit and not self.edited and not self.want_write:
            return ("You have not modified any file, so NOTHING has been fixed. Call edit_file (old_text must be copied "
                    "exactly from the file) or create_file with the complete corrected content.",
                    ["edit_file", "create_file"])
        if self.want_run and not self.ran and self.last_written:
            return (f"You have not run it yet. The user asked you to run it: call run_python now with path "
                    f"'{self.last_written}'.", ["run_python"])
        return None

    def _run_hint(self, rel: str) -> str:
        """Small models often stop after saving; nudge them to finish what was asked."""
        self.edited = True
        if rel.endswith(".py"):
            self.last_written = rel
        if self.want_run and not self.ran and rel.endswith(".py"):
            return f" NEXT STEP: the user asked you to run it, so call run_python now with path '{rel}'."
        return ""

    # ---- path safety -------------------------------------------------------------------
    def _path(self, rel) -> Path:
        if not isinstance(rel, str) or not rel.strip():
            raise CodeError("No path given.")
        rel = rel.strip().strip("\"'")
        p = Path(rel)
        if rel.count(":") > (1 if p.drive else 0):
            raise CodeError("Invalid path (alternate streams are not allowed).")
        full = (self.root / p).resolve()                 # absolute paths and '..' collapse here
        try:
            parts = full.relative_to(self.root).parts
        except ValueError:
            raise CodeError(f"Refused: '{rel}' is outside the workspace. I can only touch files inside it.") from None
        if not parts:
            raise CodeError("That is the workspace folder itself; give a file name.")
        if parts[0].lower() == ".venv":
            raise CodeError("Refused: .venv is the workspace's Python environment, managed by Jarvis. Use pip_install.")
        return full

    # ---- workspace virtual environment -------------------------------------------------
    def venv_python(self) -> Path:
        return self.venv_dir / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")

    def ensure_venv(self) -> Path:
        """Create workspace/.venv on first use and return its interpreter."""
        py = self.venv_python()
        if not py.exists():
            log.info("Creating the workspace virtual environment (first use, ~15 s)...")
            try:
                _venv.EnvBuilder(with_pip=True, clear=False, symlinks=False).create(str(self.venv_dir))
            except Exception as e:
                raise CodeError(f"Could not create the workspace virtual environment: {e}") from e
            if not py.exists():
                raise CodeError("The workspace virtual environment was created but has no python executable.")
        return py

    def _child_env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP")}
        env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        return env

    def _writable(self, rel) -> Path:
        full = self._path(rel)
        if full.suffix.lower() not in ALLOWED_EXT:
            raise CodeError(f"Refused: '{full.suffix or 'no extension'}' files are not allowed. "
                            f"Allowed: {', '.join(sorted(ALLOWED_EXT))}")
        return full

    def _existing(self, rel) -> Path:
        full = self._path(rel)
        if not full.is_file():
            raise CodeError(f"File not found in workspace: {rel}. Use list_files to see what exists.")
        return full

    def _rel(self, full: Path) -> str:
        return full.relative_to(self.root).as_posix()

    def resolve_name(self, name: str) -> str | None:
        """Forgiving lookup for spoken file names ('snake', 'Snake.py', 'snake game' -> snake_game.py)."""
        name = (name or "").strip().strip("\"'")
        if not name:
            return None
        candidates = [name, name + ".py", name.replace(" ", "_"), name.replace(" ", "_") + ".py"]
        files = {f.lower(): f for f in self._all_files()}
        for c in candidates:
            if c.lower() in files:
                return files[c.lower()]
        return None

    def _all_files(self) -> list[str]:
        out = []
        for dirpath, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", ".venv")]
            for f in files:
                out.append(Path(dirpath, f).relative_to(self.root).as_posix())
        return sorted(out)

    # ---- file operations ---------------------------------------------------------------
    def create_file(self, path, content) -> str:
        full = self._writable(path)
        if not isinstance(content, str):
            raise CodeError("content must be text.")
        if len(content.encode("utf-8")) > self.max_bytes:
            raise CodeError(f"File too large (limit {self.max_bytes // 1024} KB).")
        existed = full.exists()
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8", newline="\n")
        n = content.count("\n") + 1
        log.info("%s %s (%d lines)", "Overwrote" if existed else "Created", self._rel(full), n)
        rel = self._rel(full)
        return f"{'Overwrote' if existed else 'Created'} {rel} ({n} lines)." + self._run_hint(rel)

    def read_file(self, path) -> str:
        full = self._existing(path)
        if full.stat().st_size > self.max_bytes:
            raise CodeError("File too large to read.")
        text = full.read_text(encoding="utf-8", errors="replace")
        return text if len(text) <= 8000 else text[:8000] + "\n... [truncated]"

    def edit_file(self, path, old_text, new_text) -> str:
        full = self._writable(path)
        if not full.is_file():
            raise CodeError(f"File not found: {path}. Use create_file for new files.")
        if not isinstance(old_text, str) or not old_text or not isinstance(new_text, str):
            raise CodeError("old_text and new_text are required (old_text must not be empty).")
        text = full.read_text(encoding="utf-8")
        n = text.count(old_text)
        if n == 0:
            raise CodeError("old_text was not found. read_file the file and copy the exact text (whitespace matters).")
        if n > 1:
            raise CodeError(f"old_text matches {n} places. Include more surrounding lines so it is unique.")
        full.write_text(text.replace(old_text, new_text, 1), encoding="utf-8", newline="\n")
        rel = self._rel(full)
        log.info("Edited %s", rel)
        return f"Edited {rel}." + self._run_hint(rel)

    def list_files(self) -> str:
        files = self._all_files()
        if not files:
            return "The workspace is empty."
        return "\n".join(f"{f}  ({(self.root / f).stat().st_size} bytes)" for f in files[:200])

    def delete_file(self, path) -> str:
        full = self._existing(path)
        rel = self._rel(full)
        if not self.ui.confirm(f"Delete {rel}?"):
            return "The user declined; the file was NOT deleted."
        full.unlink()
        log.info("Deleted %s", rel)
        return f"Deleted {rel}."

    # ---- execution ---------------------------------------------------------------------
    def run_file(self, path, args: str = "") -> RunResult:
        full = self._existing(path)
        rel = self._rel(full)
        if full.suffix.lower() != ".py":
            raise CodeError("Only .py scripts can be run.")
        if self.run_failures >= self.max_failures:
            return RunResult("refused", message=(
                f"Refused: {self.max_failures} runs have already failed for this request. Stop retrying, "
                "and tell the user briefly what is going wrong."))
        try:
            argv = shlex.split(args or "")
        except ValueError:
            raise CodeError("Could not parse args.") from None

        self.ran = True
        source = full.read_text(encoding="utf-8", errors="replace")
        self.ui.show_code(rel, source)
        risks = scan_risks(source)
        if risks:
            self.ui.show_warning(rel, risks)
            log.warning("Risky operations found in %s: %s", rel, "; ".join(risks))
        question = RISKY_QUESTION if risks else f"Shall I run {rel}?"
        if not self.ui.confirm(question):
            return RunResult("declined")

        python = self.ensure_venv()                        # the workspace's own interpreter, never Jarvis's .venv
        log.info("Running %s with %s (timeout %ds, cwd=%s)...", rel, python, self.timeout, self.root)
        proc = subprocess.Popen(
            [str(python), str(full)] + argv, cwd=str(self.root), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            env=self._child_env(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            out, err = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            res = RunResult("timeout", None, out or "", err or "", message=(
                f"The script was stopped after {self.timeout}s. It may be a long-running or windowed program "
                "(a game, a server, a loop), which is not necessarily an error. Do not retry; tell the user."))
            self.ui.show_output(res)
            return res
        if proc.returncode == 0:
            res = RunResult("ok", 0, out, err)
        else:
            self.run_failures += 1
            left = self.max_failures - self.run_failures
            note = (f"Failed runs this request: {self.run_failures}/{self.max_failures}. "
                    + ("Read the traceback, fix the code, and run again." if left > 0
                       else "No attempts left: stop and tell the user what is wrong."))
            res = RunResult("failed", proc.returncode, out, err, message=note)
        self.ui.show_output(res)
        return res

    def run_python(self, path, args: str = "") -> str:
        return self.run_file(path, args).for_llm()

    def pip_install(self, package) -> str:
        if not isinstance(package, str) or not _PKG_RE.match(package.strip()):
            raise CodeError("Invalid package name (plain names with optional version pins only).")
        package = package.strip()
        if not self.ui.confirm(f"Install the {package} package with pip?"):
            return "The user declined; nothing was installed."
        python = self.ensure_venv()                        # installs go into workspace/.venv only
        log.info("pip install %s into %s ...", package, self.venv_dir)
        try:
            r = subprocess.run([str(python), "-m", "pip", "install", package], capture_output=True, text=True,
                               timeout=300, encoding="utf-8", errors="replace", cwd=str(self.root), env=self._child_env(),
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            return "pip timed out after 300 seconds."
        tail = (r.stdout + r.stderr).strip()[-1200:]
        return ("Installed " + package + "." if r.returncode == 0 else f"pip failed (exit {r.returncode}).") + "\n" + tail

    # ---- LLM dispatcher ----------------------------------------------------------------
    @staticmethod
    def _arg(args: dict, *names, default=None):
        for n in names:
            if n in args and args[n] is not None:
                return args[n]
        return default

    def call(self, name: str, args: dict) -> str:
        args = args if isinstance(args, dict) else {}
        self.tool_calls += 1
        path = self._arg(args, "path", "file_path", "filename", "file", "name")
        try:
            if name == "create_file":
                return self.create_file(path, self._arg(args, "content", "code", "contents", "text"))
            if name == "read_file":
                return self.read_file(path)
            if name == "edit_file":
                return self.edit_file(path, self._arg(args, "old_text", "old_string", "old", "search", "find"),
                                      self._arg(args, "new_text", "new_string", "new", "replace", "replacement"))
            if name == "list_files":
                return self.list_files()
            if name == "delete_file":
                return self.delete_file(path)
            if name == "run_python":
                return self.run_python(path, str(self._arg(args, "args", "arguments", default="") or ""))
            if name == "pip_install":
                return self.pip_install(self._arg(args, "package", "name", "packages"))
            return f"Unknown code tool: {name}"
        except CodeError as e:
            return f"Error: {e}"
        except OSError as e:
            return f"Error: file system problem: {e}"


# =======================================================================================
def _self_test() -> None:
    import tempfile

    class FakeUI:
        def __init__(self):
            self.answers, self.asked, self.shown, self.warned = [], [], [], []
        def confirm(self, q):
            self.asked.append(q)
            return self.answers.pop(0) if self.answers else False
        def show_code(self, rel, code): self.shown.append(rel)
        def show_warning(self, rel, findings): self.warned.append((rel, findings))
        def show_output(self, res): pass

    with tempfile.TemporaryDirectory() as d:
        ui = FakeUI()
        t = CodeTools({"code": {"workspace": d, "run_timeout_s": 3, "max_run_failures": 3}}, ui)

        assert t.call("create_file", {"path": "hello.py", "content": "print('hi')\n"}).startswith("Created hello.py")
        t.reset_turn("write hello.py and run it")
        assert "call run_python now" in t.call("create_file", {"path": "h2.py", "content": "x=1\n"})
        t.reset_turn("write hello.py")
        assert "NEXT STEP" not in t.call("create_file", {"path": "h2.py", "content": "x=1\n"})
        t.reset_turn("write a script that prints primes")
        msg, names = t.nudge()
        assert "NOTHING has been saved" in msg and names == ["create_file"]      # model only talked
        t.call("create_file", {"path": "p.py", "content": "x=1\n"})
        assert t.nudge() is None
        t.reset_turn("write p.py and run it")
        t.call("create_file", {"path": "p.py", "content": "x=1\n"})
        assert t.nudge()[1] == ["run_python"]                                    # saved but not run
        t.reset_turn("what's the weather")
        assert t.nudge() is None
        t.reset_turn("fix the bug in p.py and run it again")
        msg, names = t.nudge()
        assert "NOTHING has been fixed" in msg and names == ["edit_file", "create_file"]
        t.call("edit_file", {"path": "p.py", "old_text": "x=1", "new_text": "x=2"})
        assert t.nudge()[1] == ["run_python"]
        t.reset_turn()
        assert t.call("read_file", {"path": "hello.py"}) == "print('hi')\n"
        assert t.call("edit_file", {"path": "hello.py", "old_text": "hi", "new_text": "yo"}) == "Edited hello.py."
        assert "not found" in t.call("edit_file", {"path": "hello.py", "old_text": "zzz", "new_text": "x"})
        assert "hello.py" in t.call("list_files", {})

        # sandbox
        for bad in ["../evil.py", "sub/../../evil.py", "C:\\Windows\\evil.py", str(Path(d).parent / "evil.py"),
                    "run.exe", "x.bat", "hello.py:stream.py", "..", "."]:
            r = t.call("create_file", {"path": bad, "content": "x"})
            assert r.startswith("Error"), (bad, r)
        assert not (Path(d).parent / "evil.py").exists()
        assert t.call("read_file", {"path": "../../Windows/win.ini"}).startswith("Error")
        assert t.call("create_file", {"path": "sub/deep/a.py", "content": "x=1"}).startswith("Created")

        # confirmation gates
        ui.answers = [False]
        assert "declined" in t.call("run_python", {"path": "hello.py"}) and ui.asked[-1] == "Shall I run hello.py?"
        assert ui.shown[-1] == "hello.py", "code must be shown before the confirmation"
        ui.answers = [True]
        out = t.call("run_python", {"path": "hello.py"})
        assert "finished successfully" in out and "yo" in out, out
        ui.answers = [False]
        assert "NOT deleted" in t.call("delete_file", {"path": "hello.py"}) and (Path(d) / "hello.py").exists()
        ui.answers = [True]
        assert t.call("delete_file", {"path": "hello.py"}) == "Deleted hello.py."
        ui.answers = [False]
        assert "declined" in t.call("pip_install", {"package": "requests"})
        assert t.call("pip_install", {"package": "x; rm -rf /"}).startswith("Error")
        assert t.call("pip_install", {"package": "--index-url=http://evil x"}).startswith("Error")

        # failure budget: 3 failed runs, then refusal; timeout does not count
        t.call("create_file", {"path": "bad.py", "content": "raise ValueError('boom')\n"})
        t.reset_turn()
        for i in range(3):
            ui.answers = [True]
            r = t.call("run_python", {"path": "bad.py"})
            assert "FAILED" in r and "ValueError: boom" in r, r
        n_asked = len(ui.asked)
        ui.answers = [True]
        assert "Refused" in t.call("run_python", {"path": "bad.py"}) and len(ui.asked) == n_asked, "must not even ask"
        t.reset_turn()
        t.call("create_file", {"path": "slow.py", "content": "import time; time.sleep(30)\n"})
        ui.answers = [True]
        assert "stopped after 3s" in t.call("run_python", {"path": "slow.py"}) and t.run_failures == 0
        assert t.resolve_name("SLOW") == "slow.py"

        # ---- workspace venv: scripts run in workspace/.venv, with cwd = workspace, never in Jarvis's environment ----
        t.reset_turn()
        probe = ("import sys, os\nprint('PREFIX=' + sys.prefix)\nprint('CWD=' + os.getcwd())\n"
                 "try:\n    import ddgs\n    print('LEAK=yes')\nexcept ImportError:\n    print('LEAK=no')\n")
        t.call("create_file", {"path": "probe.py", "content": probe})
        ui.answers = [True]
        out = t.call("run_python", {"path": "probe.py"})
        assert t.venv_python().exists(), "workspace/.venv must be created on first use"
        assert f"PREFIX={t.venv_dir}" in out, out
        assert f"CWD={t.root}" in out, out
        assert "LEAK=no" in out, "Jarvis's own packages must not be importable by scripts"
        assert ".venv" not in t.call("list_files", {}) and ".venv" not in "\n".join(t._all_files())
        for p in [".venv/pyvenv.cfg", ".venv/Scripts/evil.py", "sub/../.venv/x.py"]:
            assert t.call("create_file", {"path": p, "content": "x"}).startswith("Error"), p
        assert t.call("read_file", {"path": ".venv/pyvenv.cfg"}).startswith("Error")
        assert t.call("delete_file", {"path": ".venv/pyvenv.cfg"}).startswith("Error")

        # pip_install goes through the workspace venv's interpreter (spy: no network in tests)
        calls = []
        real_run = subprocess.run
        subprocess.run = lambda argv, **kw: (calls.append(argv), subprocess.CompletedProcess(argv, 0, "ok", ""))[1]
        try:
            ui.answers = [True]
            assert t.call("pip_install", {"package": "requests==2.31.0"}).startswith("Installed")
        finally:
            subprocess.run = real_run
        assert calls and calls[0][0] == str(t.venv_python()) and calls[0][1:4] == ["-m", "pip", "install"], calls
        assert str(sys.executable) != calls[0][0]

        # ---- risk scan + spoken question ----
        risky = {
            "import os\nos.remove('x.txt')\n": "deletes files",
            "import shutil\nshutil.rmtree('d')\n": "deletes files",
            "from pathlib import Path\nPath('a').unlink()\n": "deletes files",
            "open('C:/Users/me/secret.txt', 'w').write('x')": "outside the workspace",
            "open('/etc/passwd')": "outside the workspace",
            "open('../up.txt', 'a')": "outside the workspace",
            "p = input_path\nopen(p, 'w').write('x')": "computed path",
            "import subprocess\nsubprocess.run(['dir'])\n": "runs other programs",
            "import os\nos.system('dir')\n": "runs other programs",
            "import requests\nrequests.get('http://x')\n": "uses the network",
            "import socket\n": "uses the network",
            "import winreg\n": "registry",
            "x = 'HKEY_LOCAL_MACHINE'\n": "registry",
            "eval(input_text)\n": "dynamically built code",
        }
        for src, expect in risky.items():
            found = scan_risks(src)
            assert found and any(expect in f for f in found), (src, found)
        safe = ["print('hi')\n", "import math, json, random\nopen('out.txt','w').write('x')\nopen('data.csv').read()\n",
                "from pathlib import Path\nPath('notes.txt').write_text('x')\n", "def f(:\n"[:0] + "x = [i for i in range(5)]\n"]
        for src in safe:
            assert scan_risks(src) == [], (src, scan_risks(src))
        assert any("line 2" in f for f in scan_risks("import os\nos.remove('x')\n"))

        t.reset_turn()
        t.call("create_file", {"path": "risky.py", "content": "import os\nos.remove('nothing.txt')\n"})
        t.call("create_file", {"path": "calm.py", "content": "print('calm')\n"})
        ui.answers, ui.warned = [False], []
        assert "declined" in t.call("run_python", {"path": "risky.py"})
        assert ui.asked[-1] == RISKY_QUESTION and ui.warned and ui.warned[0][0] == "risky.py", "risky -> special question"
        ui.answers, ui.warned = [False], []
        t.call("run_python", {"path": "calm.py"})
        assert ui.asked[-1] == "Shall I run calm.py?" and not ui.warned, "benign -> normal question, no warning"
    print("OK: code tool tests passed")


if __name__ == "__main__":
    _self_test()
