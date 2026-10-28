"""Tool registry: schemas offered to the LLM and a dispatcher that runs them.

web_search is always offered. The coding tools (create_file, read_file, edit_file, list_files, delete_file,
run_python, pip_install) are offered only on coding turns, once main.py has called configure().
"""
import json
import logging

from tools import code, web_search

log = logging.getLogger("jarvis.tools")

_code: "code.CodeTools | None" = None


def configure(code_tools: "code.CodeTools") -> None:
    global _code
    _code = code_tools


def schemas(coding: bool = False) -> list[dict]:
    out = [web_search.SCHEMA]
    if coding and _code is not None:
        out += code.SCHEMAS
    return out


def dispatch(name: str, arguments: dict, cfg: dict) -> str:
    """Run a tool and always return a string for the LLM (errors included)."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    arguments = arguments or {}
    try:
        if name == "web_search":
            return web_search.run(arguments.get("query", ""), cfg.get("web_search", {}))
        if name in code.TOOL_NAMES:
            if _code is None:
                return "Coding tools are not available."
            return _code.call(name, arguments)
        return f"Unknown tool: {name}"
    except Exception as e:  # tool failures must never crash the assistant
        log.exception("tool %s failed", name)
        return f"Tool {name} failed: {e}"
