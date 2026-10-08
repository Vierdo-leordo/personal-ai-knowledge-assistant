"""Persistence + Memory Layer (SQLite).

Stores: the document registry, conversations & messages (questions, responses, retrieved sources),
user preferences, frequently accessed topics, and saved study outputs.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .embeddings import tokenize

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_type TEXT,
    category TEXT DEFAULT 'General',
    content_hash TEXT UNIQUE,
    num_pages INTEGER,
    num_chunks INTEGER,
    word_count INTEGER,
    size_bytes INTEGER,
    stored_path TEXT,
    upload_date TEXT,
    embedder TEXT
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT,
    content TEXT,
    sources TEXT,
    metadata TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS topics (
    topic TEXT PRIMARY KEY,
    count INTEGER DEFAULT 0,
    last_accessed TEXT
);
CREATE TABLE IF NOT EXISTS study_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT,
    title TEXT,
    content TEXT,
    created_at TEXT
);
"""

DEFAULT_PREFERENCES = {
    "answer_style": "Balanced",  # Concise | Balanced | Detailed
    "expertise_level": "Student",  # Beginner | Student | Expert
    "language": "English",
    "custom_instructions": "",
    "voice_replies": "false",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with self._lock:
                yield conn
                conn.commit()
        finally:
            conn.close()

    # ---------------------------------------------------------------- documents
    def add_document(self, **doc) -> None:
        cols = ",".join(doc.keys())
        qs = ",".join("?" for _ in doc)
        with self._conn() as c:
            c.execute(f"INSERT OR REPLACE INTO documents ({cols}) VALUES ({qs})", tuple(doc.values()))

    def get_document_by_hash(self, content_hash: str) -> Optional[Dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM documents WHERE content_hash=?", (content_hash,)).fetchone()
        return dict(row) if row else None

    def get_document(self, doc_id: str) -> Optional[Dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self, category: Optional[str] = None) -> List[Dict]:
        q, args = "SELECT * FROM documents", ()
        if category and category != "All":
            q, args = q + " WHERE category=?", (category,)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY upload_date DESC", args).fetchall()]

    def update_document_category(self, doc_id: str, category: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE documents SET category=? WHERE doc_id=?", (category, doc_id))

    def delete_document(self, doc_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))

    # ---------------------------------------------------------------- conversations
    def create_conversation(self, title: str = "New conversation") -> str:
        cid = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute("INSERT INTO conversations VALUES (?,?,?,?)", (cid, title, _now(), _now()))
        return cid

    def list_conversations(self) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT cv.*, COUNT(m.id) AS n_messages FROM conversations cv
                   LEFT JOIN messages m ON m.conversation_id = cv.id
                   GROUP BY cv.id ORDER BY cv.updated_at DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, cid: str, title: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE conversations SET title=? WHERE id=?", (title, cid))

    def delete_conversation(self, cid: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
            c.execute("DELETE FROM conversations WHERE id=?", (cid,))

    def add_message(
        self, cid: str, role: str, content: str, sources: Optional[List[Dict]] = None, metadata: Optional[Dict] = None
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO messages (conversation_id, role, content, sources, metadata, created_at) VALUES (?,?,?,?,?,?)",
                (cid, role, content, json.dumps(sources or []), json.dumps(metadata or {}), _now()),
            )
            c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), cid))

    def get_messages(self, cid: str, limit: Optional[int] = None) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (cid,)
            ).fetchall()
        msgs = []
        for r in rows:
            d = dict(r)
            d["sources"] = json.loads(d["sources"] or "[]")
            d["metadata"] = json.loads(d["metadata"] or "{}")
            msgs.append(d)
        return msgs[-limit:] if limit else msgs

    def search_messages(self, text: str) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT m.*, cv.title FROM messages m JOIN conversations cv ON cv.id = m.conversation_id
                   WHERE m.content LIKE ? ORDER BY m.id DESC LIMIT 50""",
                (f"%{text}%",),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- preferences
    def get_preferences(self) -> Dict[str, str]:
        prefs = dict(DEFAULT_PREFERENCES)
        with self._conn() as c:
            for r in c.execute("SELECT key, value FROM preferences").fetchall():
                prefs[r["key"]] = r["value"]
        return prefs

    def set_preference(self, key: str, value: Any) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO preferences VALUES (?,?)", (key, str(value)))

    def preferences_prompt(self) -> str:
        p = self.get_preferences()
        style = {
            "Concise": "Keep answers short: 2-4 sentences or a few bullets.",
            "Balanced": "Give a clear answer with moderate detail.",
            "Detailed": "Give thorough, detailed explanations with examples.",
        }.get(p["answer_style"], "")
        level = {
            "Beginner": "Explain like the user is new to the subject; avoid jargon or define it.",
            "Student": "Pitch explanations at a university-student level.",
            "Expert": "The user is an expert; be precise and technical.",
        }.get(p["expertise_level"], "")
        lines = ["\nUser preferences:", f"- {style}", f"- {level}", f"- Respond in {p['language']}."]
        if p.get("custom_instructions"):
            lines.append(f"- {p['custom_instructions']}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- topics
    def record_topics(self, text: str, max_topics: int = 4) -> List[str]:
        toks = [t for t in tokenize(text) if len(t) > 3]
        topics = list(dict.fromkeys(toks))[:max_topics]
        with self._conn() as c:
            for t in topics:
                c.execute(
                    """INSERT INTO topics (topic, count, last_accessed) VALUES (?, 1, ?)
                       ON CONFLICT(topic) DO UPDATE SET count = count + 1, last_accessed = excluded.last_accessed""",
                    (t, _now()),
                )
        return topics

    def top_topics(self, n: int = 10) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM topics ORDER BY count DESC, last_accessed DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in rows]

    def clear_topics(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM topics")

    # ---------------------------------------------------------------- study outputs
    def save_output(self, kind: str, title: str, content: str) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO study_outputs (kind, title, content, created_at) VALUES (?,?,?,?)",
                (kind, title, content, _now()),
            )
            return int(cur.lastrowid)

    def list_outputs(self, kind: Optional[str] = None) -> List[Dict]:
        q, args = "SELECT * FROM study_outputs", ()
        if kind:
            q, args = q + " WHERE kind=?", (kind,)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY id DESC", args).fetchall()]

    def delete_output(self, output_id: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM study_outputs WHERE id=?", (output_id,))
