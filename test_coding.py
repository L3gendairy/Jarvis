"""End-to-end coding test with the real LLMs, real workspace and the real confirmation flow.

Run:  python test_coding.py
  A. "write fib.py (first 20 Fibonacci numbers) and run it" -> coder model used, file saved, run was confirmed,
     output verified, nothing code-like was spoken.
  B. The user says NO -> the script is not run.
  C. A broken script is run via "run buggy.py" -> it fails, Jarvis fixes it and re-runs (max 3 failed attempts).
Speakers are muted during the test (replies are only printed).
"""
import re
import shutil
import subprocess
import sys
import threading
import time

from common import load_config, resolve, setup_logging
from main import Jarvis

WS = "data/coding_test_ws"
FIB20 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise SystemExit(1)


def main() -> None:
    setup_logging("INFO")
    shutil.rmtree(resolve(WS), ignore_errors=True)
    cfg = load_config()
    cfg["code"]["workspace"] = WS
    cfg["settings_file"] = "data/coding_test_settings.json"
    cfg["memory"]["db_path"] = "data/coding_test_memory.db"
    j = Jarvis(cfg, text_only=True)
    j.llm.ensure_server()
    j.llm.ensure_models()
    j.tts._muted = True                       # keep the speakers quiet; text is still recorded below

    spoken, questions, run_status = [], [], []
    orig_say, orig_confirm, orig_run = j.tts.say, j.confirm, j.code.run_file
    j.tts.say = lambda t: (spoken.append(t), orig_say(t))[0]
    j.confirm = lambda q: (questions.append(q), orig_confirm(q))[1]

    def spy_run(*a, **kw):
        res = orig_run(*a, **kw)
        run_status.append(res.status)
        return res
    j.code.run_file = spy_run

    answer = {"text": "y"}
    stop = threading.Event()

    def answerer() -> None:                   # plays the user at the confirmation prompt
        while not stop.is_set():
            if j.confirming.is_set():
                time.sleep(0.4)
                j.events.put(("text", answer["text"], "en"))
                while j.confirming.is_set() and not stop.is_set():
                    time.sleep(0.05)
            time.sleep(0.05)
    threading.Thread(target=answerer, daemon=True).start()

    # ---- A -----------------------------------------------------------------------------
    print("\n[A] Write fib.py and run it")
    j.handle("text", "Jarvis, write a Python script called fib.py that prints the first 20 Fibonacci numbers, and run it.", "en")
    check(j.llm.last_model == j.llm.coder_model, f"coding request used the coder model ({j.llm.last_model})")
    fib = resolve(WS) / "fib.py"
    check(fib.exists(), "fib.py was saved in the workspace")
    check(any("run" in q.lower() and "fib.py" in q for q in questions), f"asked for confirmation: {questions}")
    check("ok" in run_status, f"script ran successfully (run statuses: {run_status})")
    out = subprocess.run([sys.executable, str(fib)], capture_output=True, text=True, timeout=30).stdout
    nums = [int(x) for x in re.findall(r"\d+", out)]
    check(nums[:20] == FIB20, f"output contains the first 20 Fibonacci numbers: {nums[:20]}")
    check(not any("def " in s or "print(" in s or "```" in s for s in spoken), "no code was spoken aloud")
    print("  spoken summary:", " | ".join(s for s in spoken if s not in questions))

    # ---- B -----------------------------------------------------------------------------
    print("\n[B] User says no")
    answer["text"] = "n"
    run_status.clear()
    j.handle("text", "Write a script called hello.py that prints hello, and run it.", "en")
    check((resolve(WS) / "hello.py").exists(), "hello.py was saved")
    check("ok" not in run_status and "failed" not in run_status, f"declined: script was not run ({run_status})")

    # ---- C -----------------------------------------------------------------------------
    print("\n[C] Failing script -> read error, fix, retry")
    answer["text"] = "y"
    run_status.clear()
    (resolve(WS) / "buggy.py").write_text("total = sum(rang(10))\nprint('total is', total)\n", encoding="utf-8")
    j.handle("text", "run buggy.py", "en")
    check(run_status and run_status[0] == "failed", f"first run failed as expected ({run_status})")
    check("ok" in run_status, f"Jarvis fixed it and a later run succeeded ({run_status})")
    check(run_status.count("failed") <= 3, "no more than 3 failed attempts")
    check("rang(" not in (resolve(WS) / "buggy.py").read_text(encoding="utf-8"), "the typo was fixed in the file")

    stop.set()
    j.close()
    j.memory.close()
    shutil.rmtree(resolve(WS), ignore_errors=True)
    for f in ("data/coding_test_settings.json", "data/coding_test_memory.db"):
        resolve(f).unlink(missing_ok=True)
    print("\nALL CODING TESTS PASSED")


if __name__ == "__main__":
    main()
