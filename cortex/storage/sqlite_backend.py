"""
sqlite_backend.py — local-first SQLite backend for Cortex.

Lets Cortex run with NO network: the agent's own data (context, memory,
reminders, goals, priorities, autonomy log, experiences) lives in a local
SQLite file, so everything works with Tailscale off (e.g. on a college network
that blocks VPNs). The remote Supabase/Postgres becomes an optional SYNC target
(see sync.py), not a hard dependency.

This module is a thin psycopg2-compatibility shim so `db.py`'s existing query
code runs almost unchanged against SQLite:
  * `%s` placeholders are rewritten to `?`
  * `NOW()` is rewritten to `CURRENT_TIMESTAMP`
  * TIMESTAMP columns read back as `datetime` (like psycopg2), JSON columns read
    back as parsed Python objects (like JSONB)
  * rows are `sqlite3.Row` → support both `row[0]` and `row["col"]` and `dict(row)`

pgvector operations (`<=>`) have no SQLite equivalent and are handled in db.py
by ranking embeddings in Python (cosine_rank below) — fine at a personal
agent's data scale.
"""
from __future__ import annotations

import os
import re
import json
import math
import sqlite3
import logging
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger("cortex.db.sqlite")

# ── type adapters/converters: match psycopg2's return shapes ───────────────
# Store datetimes in the SAME textual format CURRENT_TIMESTAMP uses
# ('YYYY-MM-DD HH:MM:SS') so lexicographic comparison in WHERE clauses is
# correct regardless of which side wrote the value.
sqlite3.register_adapter(datetime, lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S"))


def _parse_ts(b: bytes):
    s = b.decode()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return s  # leave as string if an unexpected format slips in


def _parse_json(b: bytes):
    if b is None:
        return None
    try:
        return json.loads(b)
    except (ValueError, TypeError):
        return b.decode() if isinstance(b, bytes) else b


sqlite3.register_converter("TIMESTAMP", _parse_ts)
sqlite3.register_converter("JSON", _parse_json)

# ── SQL translation (the PG→SQLite dialect delta db.py actually uses) ───────
_NOW_RE = re.compile(r"\bNOW\(\)", re.IGNORECASE)


def translate(sql: str) -> str:
    sql = sql.replace("%s", "?")
    sql = _NOW_RE.sub("CURRENT_TIMESTAMP", sql)
    return sql


# ── cosine ranking for the vector methods (pgvector substitute) ────────────
def cosine_rank(query_vec, rows, vec_key: str, limit: int):
    """
    rows: list of dict-like, each with rows[vec_key] = JSON string of a float list
    (or a list). Returns the top `limit` rows by cosine similarity, each with a
    "_similarity" float added, highest first.
    """
    if not query_vec:
        return []
    q = [float(x) for x in query_vec]
    qn = math.sqrt(sum(x * x for x in q)) or 1.0
    scored = []
    for r in rows:
        raw = r[vec_key]
        if raw is None:
            continue
        vec = raw if isinstance(raw, list) else json.loads(raw)
        if not vec:
            continue
        dot = sum(a * float(b) for a, b in zip(q, vec))
        vn = math.sqrt(sum(float(b) * float(b) for b in vec)) or 1.0
        scored.append((dot / (qn * vn), r))
    scored.sort(key=lambda t: t[0], reverse=True)
    out = []
    for sim, r in scored[:limit]:
        d = dict(r)
        d["_similarity"] = sim
        out.append(d)
    return out


# ── psycopg2-shaped wrappers over sqlite3 ──────────────────────────────────
class _Cursor:
    def __init__(self, conn: sqlite3.Connection):
        self._cur = conn.cursor()

    def execute(self, sql, params=()):
        self._cur.execute(translate(sql), tuple(params) if params else ())
        return self

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    @property
    def rowcount(self):
        return self._cur.rowcount

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cur.close()
        return False


class _Conn:
    """Mimics the psycopg2 connection surface db.py uses (cursor/commit/rollback)."""

    def __init__(self, sqlite_conn: sqlite3.Connection):
        self._c = sqlite_conn

    def cursor(self, cursor_factory=None):  # factory ignored; Row covers both uses
        return _Cursor(self._c)

    def commit(self):
        self._c.commit()

    def rollback(self):
        self._c.rollback()


# ── schema ─────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS context_store (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    project_path TEXT,
    key TEXT NOT NULL,
    value JSON NOT NULL,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    embedding TEXT,
    UNIQUE(category, project_path, key)
);
CREATE INDEX IF NOT EXISTS idx_context_category ON context_store(category);
CREATE INDEX IF NOT EXISTS idx_context_collected ON context_store(collected_at DESC);

CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    importance INTEGER DEFAULT 3,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    embedding TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(type);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tokens_used INTEGER,
    model_used TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    due_at TIMESTAMP NOT NULL,
    repeat_rule TEXT,
    context_snapshot JSON,
    fired INTEGER DEFAULT 0,
    fired_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_at);

CREATE TABLE IF NOT EXISTS command_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command TEXT NOT NULL,
    exit_code INTEGER,
    stdout TEXT,
    stderr TEXT,
    duration_ms INTEGER,
    triggered_by TEXT,
    project_path TEXT,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tracked_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    type TEXT,
    collect_git INTEGER DEFAULT 1,
    collect_deps INTEGER DEFAULT 1,
    custom_commands JSON,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS devops_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT NOT NULL UNIQUE,
    task TEXT NOT NULL,
    project_path TEXT,
    steps JSON NOT NULL,
    status TEXT DEFAULT 'pending',
    current_step INTEGER DEFAULT 0,
    results JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_plans_planid ON devops_plans(plan_id);

CREATE TABLE IF NOT EXISTS experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_text TEXT NOT NULL,
    state_context JSON,
    action TEXT NOT NULL,
    action_payload JSON NOT NULL,
    speak TEXT,
    exit_code INTEGER,
    was_corrected INTEGER DEFAULT 0,
    was_confirmed INTEGER,
    explicit_feedback INTEGER,
    reward REAL,
    reward_reason TEXT,
    embedding TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reward_computed_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_experiences_session ON experiences(session_id);
CREATE INDEX IF NOT EXISTS idx_experiences_reward ON experiences(reward DESC);

CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    priority INTEGER DEFAULT 3,
    source TEXT DEFAULT 'self',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_evaluated_at TIMESTAMP,
    completed_at TIMESTAMP,
    result TEXT
);
CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);

CREATE TABLE IF NOT EXISTS autonomy_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    think TEXT,
    decision TEXT,
    goal_id INTEGER,
    action_payload JSON,
    needs_confirmation INTEGER DEFAULT 0,
    confirmed INTEGER,
    exit_code INTEGER,
    notified INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_autonomy_log_time ON autonomy_log(cycle_at DESC);

CREATE TABLE IF NOT EXISTS priorities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    statement TEXT NOT NULL,
    weight INTEGER DEFAULT 3,
    source TEXT DEFAULT 'user',
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_reinforced_at TIMESTAMP,
    reinforcement_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_priorities_status ON priorities(status);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    chunk TEXT NOT NULL,
    embedding TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);

CREATE TABLE IF NOT EXISTS nudges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    body TEXT,
    kind TEXT DEFAULT 'info',
    source TEXT DEFAULT 'autonomy',
    read INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_nudges_read ON nudges(read);
"""


def open_db(path: str) -> sqlite3.Connection:
    """Open (and initialize) the local SQLite DB."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(
        path,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
        timeout=15.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")      # concurrent readers, durable
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=15000;")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn
