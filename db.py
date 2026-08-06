"""
Database layer — PostgreSQL via Supabase pooler.
Connection pool + helpers for all Cortex data operations.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool, extras

logger = logging.getLogger("cortex.db")


class Database:
    """PostgreSQL connection manager with pooling."""

    def __init__(self, host: str, port: int, name: str, user: str, password: str):
        self.host = host
        self.port = port
        self.name = name
        self.user = user
        self.password = password or os.environ.get("CORTEX_DB_PASSWORD", "")
        self._pool = None

    def connect(self):
        """Initialize connection pool."""
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                host=self.host,
                port=self.port,
                dbname=self.name,
                user=self.user,
                password=self.password,
                connect_timeout=10
            )
            logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}/{self.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to DB: {e}")
            return False

    def close(self):
        """Close all pool connections."""
        if self._pool:
            self._pool.closeall()
            logger.info("Database pool closed")

    @contextmanager
    def get_conn(self):
        """Get a connection from the pool (context manager)."""
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    # ─── Context Store ───

    def upsert_context(self, category: str, key: str, value: dict,
                       project_path: str = None, expires_minutes: int = 10):
        """Insert or update a context entry (collectors write here)."""
        expires_at = datetime.utcnow() + timedelta(minutes=expires_minutes)
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO context_store (category, project_path, key, value, collected_at, expires_at)
                    VALUES (%s, %s, %s, %s, NOW(), %s)
                    ON CONFLICT (category, project_path, key)
                    DO UPDATE SET value = EXCLUDED.value,
                                  collected_at = NOW(),
                                  expires_at = EXCLUDED.expires_at
                """, (category, project_path, key, json.dumps(value), expires_at))

    def get_context(self, categories: list = None, project_path: str = None,
                    max_age_minutes: int = 10) -> list:
        """Read pre-computed context from DB."""
        # DB timestamps are in UTC (Supabase default)
        from datetime import timezone
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=max_age_minutes)
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                query = """
                    SELECT category, key, value, collected_at
                    FROM context_store
                    WHERE collected_at > %s
                """
                params = [cutoff]

                if categories:
                    query += " AND category = ANY(%s)"
                    params.append(categories)

                if project_path:
                    query += " AND (project_path = %s OR project_path IS NULL)"
                    params.append(project_path)

                query += " ORDER BY category, collected_at DESC"
                cur.execute(query, params)
                rows = cur.fetchall()

        return [{"category": r["category"], "key": r["key"],
                 "value": r["value"], "collected_at": str(r["collected_at"])}
                for r in rows]

    # ─── Command Log ───

    def log_command(self, command: str, exit_code: int, stdout: str, stderr: str,
                    duration_ms: int, triggered_by: str, project_path: str = None):
        """Log an executed command."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO command_log (command, exit_code, stdout, stderr, duration_ms, triggered_by, project_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (command, exit_code, stdout, stderr, duration_ms, triggered_by, project_path))

    # ─── Conversations ───

    def save_conversation(self, session_id: str, role: str, content: str,
                          tokens: int = None, model: str = None):
        """Store a conversation turn."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversations (session_id, role, content, tokens_used, model_used)
                    VALUES (%s, %s, %s, %s, %s)
                """, (session_id, role, content, tokens, model))

    def get_conversations(self, session_id: str, limit: int = 10) -> list:
        """Get recent conversation history for a session."""
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT role, content, timestamp
                    FROM conversations
                    WHERE session_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (session_id, limit))
                rows = cur.fetchall()

        # Return in chronological order
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # ─── Memory ───

    def save_memory(self, type: str, content: str, source: str = None, importance: int = 3):
        """Store a memory fact."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO memory (type, content, source, importance)
                    VALUES (%s, %s, %s, %s)
                """, (type, content, source, importance))

    def get_memories(self, type: str = None, limit: int = 20) -> list:
        """Retrieve memories, ordered by importance."""
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                query = "SELECT type, content, importance FROM memory"
                params = []

                if type:
                    query += " WHERE type = %s"
                    params.append(type)

                query += " ORDER BY importance DESC, created_at DESC LIMIT %s"
                params.append(limit)

                cur.execute(query, params)
                rows = cur.fetchall()

        return [{"type": r["type"], "content": r["content"], "importance": r["importance"]}
                for r in rows]

    # ─── Reminders ───

    def get_pending_reminders(self) -> list:
        """Get reminders that are due (compares against provided local time)."""
        now_local = datetime.now()
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, text, due_at, repeat_rule
                    FROM reminders
                    WHERE fired = FALSE AND due_at <= %s
                    ORDER BY due_at ASC
                """, (now_local,))
                return cur.fetchall()

    def create_reminder(self, text: str, due_at: datetime, repeat_rule: str = None) -> int:
        """Create a new reminder. Returns its ID."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO reminders (text, due_at, repeat_rule)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (text, due_at, repeat_rule))
                return cur.fetchone()[0]

    def mark_reminder_fired(self, reminder_id: int):
        """Mark a reminder as fired."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE reminders SET fired = TRUE, fired_at = NOW()
                    WHERE id = %s
                """, (reminder_id,))

    # ─── DevOps Plans ───

    def save_plan(self, plan_id: str, task: str, steps: list, project_path: str = None):
        """Save a new devops plan."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO devops_plans (plan_id, task, project_path, steps)
                    VALUES (%s, %s, %s, %s)
                """, (plan_id, task, project_path, json.dumps(steps)))

    def get_plan(self, plan_id: str) -> dict:
        """Get a plan by its ID."""
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM devops_plans WHERE plan_id = %s", (plan_id,))
                return cur.fetchone()

    def update_plan_status(self, plan_id: str, status: str, current_step: int = None, results: list = None):
        """Update plan execution status."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                updates = ["status = %s"]
                params = [status]
                if current_step is not None:
                    updates.append("current_step = %s")
                    params.append(current_step)
                if results is not None:
                    updates.append("results = %s")
                    params.append(json.dumps(results))
                if status in ("completed", "failed"):
                    updates.append("completed_at = NOW()")

                params.append(plan_id)
                cur.execute(
                    f"UPDATE devops_plans SET {', '.join(updates)} WHERE plan_id = %s",
                    params
                )

    # ─── Tracked Projects ───

    def get_tracked_projects(self) -> list:
        """Get all tracked projects."""
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("SELECT name, path, type, custom_commands FROM tracked_projects")
                return cur.fetchall()

    def track_project(self, name: str, path: str, project_type: str = None, custom_commands: dict = None):
        """Register a new tracked project."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO tracked_projects (name, path, type, custom_commands)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (path)
                    DO UPDATE SET name = EXCLUDED.name,
                                  type = EXCLUDED.type,
                                  custom_commands = EXCLUDED.custom_commands
                """, (name, path, project_type, json.dumps(custom_commands) if custom_commands else None))

    # ─── Maintenance ───

    def purge_expired_context(self):
        """Delete context rows past their expires_at."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM context_store WHERE expires_at < NOW()")
                deleted = cur.rowcount
        if deleted:
            logger.info(f"Purged {deleted} expired context rows")
        return deleted

    def health_check(self) -> bool:
        """Quick DB connectivity check."""
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception:
            return False
