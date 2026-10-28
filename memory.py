"""Long-term memory (SQLite): facts about the user, preferences, things to remember.

Relevance is keyword overlap plus recency; the store is small so no vector DB is needed.

Standalone test:  python memory.py
"""
import json
import logging
import re
import sqlite3
import threading
import time
from difflib import SequenceMatcher

from common import normalize

log = logging.getLogger("jarvis.memory")

_STOP = set("""a an the and or but is are was were be been am i me my you your he she it we they of to in on at for
with about from by as that this these those do does did have has had what who how when where which can could would
should will just not no yes please tell know like le la les un une des et ou mais est suis es sont de du en dans sur
pour avec que qui quoi je tu il elle nous vous ils mon ma mes ton ta tes""".split())

EXTRACT_PROMPT = """You extract durable personal facts about the USER from a conversation turn.
Return ONLY a JSON object: {"facts": ["...", "..."]}.
Rules:
- Only stable facts worth remembering later: name, job, location, family, pets, preferences, habits, projects, goals, important dates.
- Write each fact as a short third-person statement, e.g. "The user's dog is called Max." or "The user prefers dark roast coffee."
- Do NOT include: questions, temporary states, small talk, things the assistant said, or facts about the world.
- If there is nothing worth remembering, return {"facts": []}.

User said: %s
Assistant replied: %s"""


def _tokens(text: str) -> set[str]:
    return {t for t in normalize(text).split() if t not in _STOP and len(t) > 1}


class Memory:
    def __init__(self, db_path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS facts (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       text TEXT NOT NULL,
                       source TEXT NOT NULL DEFAULT 'auto',
                       created REAL NOT NULL,
                       last_used REAL
                   )"""
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---- writes ------------------------------------------------------------------------
    def add(self, text: str, source: str = "user") -> tuple[int | None, bool]:
        """Store a fact. Returns (id, created). Near-duplicates are not stored twice."""
        text = text.strip().rstrip()
        if not text:
            return None, False
        with self._lock:
            norm = normalize(text)
            for row in self._db.execute("SELECT id, text FROM facts"):
                other = normalize(row["text"])
                if norm == other or SequenceMatcher(None, norm, other).ratio() > 0.85:
                    return row["id"], False
            cur = self._db.execute(
                "INSERT INTO facts (text, source, created) VALUES (?, ?, ?)", (text, source, time.time())
            )
            self._db.commit()
            return cur.lastrowid, True

    def forget(self, query: str) -> list[str]:
        """Delete facts matching the query (all keywords must be present, else best overlap)."""
        q = _tokens(query)
        if not q:
            return []
        with self._lock:
            rows = list(self._db.execute("SELECT id, text FROM facts"))
            scored = [(len(q & _tokens(r["text"])) / len(q), r) for r in rows]
            full = [r for s, r in scored if s >= 1.0]
            hits = full or [r for s, r in scored if s >= 0.5]
            for r in hits:
                self._db.execute("DELETE FROM facts WHERE id = ?", (r["id"],))
            self._db.commit()
            return [r["text"] for r in hits]

    def forget_all(self) -> int:
        with self._lock:
            n = self._db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            self._db.execute("DELETE FROM facts")
            self._db.commit()
            return n

    # ---- reads -------------------------------------------------------------------------
    def all(self) -> list[str]:
        with self._lock:
            return [r["text"] for r in self._db.execute("SELECT text FROM facts ORDER BY created")]

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def relevant(self, query: str, limit: int = 8) -> list[str]:
        """Facts most relevant to the query; padded with the newest facts when few match."""
        q = _tokens(query)
        with self._lock:
            rows = list(self._db.execute("SELECT id, text, created FROM facts"))
        if not rows:
            return []
        newest = max(r["created"] for r in rows)
        oldest = min(r["created"] for r in rows)
        span = max(newest - oldest, 1.0)
        scored = []
        for r in rows:
            overlap = len(q & _tokens(r["text"]))
            recency = (r["created"] - oldest) / span          # 0..1
            scored.append((overlap * 2 + recency * 0.5, overlap, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [r for _, ov, r in scored if ov > 0][:limit]
        for _, _, r in scored:                                  # pad with newest facts
            if len(chosen) >= min(limit, 4):
                break
            if r not in chosen:
                chosen.append(r)
        now = time.time()
        with self._lock:
            for r in chosen:
                self._db.execute("UPDATE facts SET last_used = ? WHERE id = ?", (now, r["id"]))
            self._db.commit()
        return [r["text"] for r in chosen]

    # ---- auto extraction ---------------------------------------------------------------
    def extract_and_store(self, llm, user_text: str, reply: str) -> list[str]:
        """Ask a (light) LLM call for durable facts in this turn and store the new ones."""
        if len(user_text.split()) < 3:
            return []
        raw = llm.quick_json(EXTRACT_PROMPT % (user_text, reply))
        facts = []
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            facts = [f for f in data.get("facts", []) if isinstance(f, str)]
        except (json.JSONDecodeError, AttributeError):
            m = re.search(r"\{.*\}", raw or "", re.S)
            if m:
                try:
                    facts = [f for f in json.loads(m.group()).get("facts", []) if isinstance(f, str)]
                except json.JSONDecodeError:
                    pass
        added = []
        for f in facts[:5]:
            _, created = self.add(f, source="auto")
            if created:
                added.append(f)
        return added


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        m = Memory(Path(d) / "m.db")
        m.add("The user's dog is called Max.", "user")
        m.add("The user prefers dark roast coffee.", "user")
        m.add("The user works as a game developer.", "user")
        assert m.add("The user's dog is called Max", "auto")[1] is False, "duplicate stored"
        assert m.count() == 3
        assert "Max" in m.relevant("what is my dog's name?")[0]
        assert m.forget("coffee") == ["The user prefers dark roast coffee."]
        assert m.count() == 2
        print("OK: memory tests passed; facts:", m.all())
        m.close()
