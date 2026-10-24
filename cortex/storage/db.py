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

    def __init__(self, host: str = None, port: int = None, name: str = None,
                 user: str = None, password: str = None,
                 backend: str = "postgres", sqlite_path: str = None):
        self.backend = backend
        self.host = host
        self.port = port
        self.name = name
        self.user = user
        self.password = password or os.environ.get("CORTEX_DB_PASSWORD", "")
        self.sqlite_path = sqlite_path or os.path.expanduser("~/.cortex/cortex.db")
        self._pool = None
        self._sqlite = None  # the raw sqlite3 connection (local-first backend)

    def connect(self):
        """Initialize the backend. SQLite = local-first (works offline)."""
        if self.backend == "sqlite":
            try:
                import cortex.storage.sqlite_backend as sqlite_backend
                self._sqlite = sqlite_backend.open_db(self.sqlite_path)
                logger.info(f"Connected to local SQLite at {self.sqlite_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to open SQLite DB: {e}")
                return False
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=2,
                # threaded=True on the Flask server plus background executors
                # (memory extraction, experience logging, autonomy) now hold
                # connections concurrently — 10 was tuned for one-request-at-a-time.
                maxconn=20,
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
        """Close the backend."""
        if self._sqlite:
            self._sqlite.close()
            self._sqlite = None
        if self._pool:
            self._pool.closeall()
            logger.info("Database pool closed")

    @contextmanager
    def get_conn(self):
        """Get a connection (context manager). Commits on success, rolls back on error."""
        if self.backend == "sqlite":
            import cortex.storage.sqlite_backend as sqlite_backend
            conn = sqlite_backend._Conn(self._sqlite)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return
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
                       project_path: str = None, expires_minutes: int = 10,
                       embedding_text: str = None):
        """
        Insert or update a context entry (collectors write here).
        embedding_text: optional text to embed for semantic retrieval. If provided,
        this row becomes searchable via search_context_semantic().
        """
        expires_at = datetime.utcnow() + timedelta(minutes=expires_minutes)
        vec_literal = None
        if embedding_text:
            try:
                from cortex.llm.embeddings import embed, to_pgvector
                vec_literal = to_pgvector(embed(embedding_text))
            except Exception as e:
                logger.warning(f"Context embedding failed for {category}/{key}: {e}")

        vec_cast = "::vector" if self.backend == "postgres" else ""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                if vec_literal:
                    cur.execute(f"""
                        INSERT INTO context_store (category, project_path, key, value, collected_at, expires_at, embedding)
                        VALUES (%s, %s, %s, %s, NOW(), %s, %s{vec_cast})
                        ON CONFLICT (category, project_path, key)
                        DO UPDATE SET value = EXCLUDED.value,
                                      collected_at = NOW(),
                                      expires_at = EXCLUDED.expires_at,
                                      embedding = EXCLUDED.embedding
                    """, (category, project_path, key, json.dumps(value), expires_at, vec_literal))
                else:
                    cur.execute("""
                        INSERT INTO context_store (category, project_path, key, value, collected_at, expires_at)
                        VALUES (%s, %s, %s, %s, NOW(), %s)
                        ON CONFLICT (category, project_path, key)
                        DO UPDATE SET value = EXCLUDED.value,
                                      collected_at = NOW(),
                                      expires_at = EXCLUDED.expires_at
                    """, (category, project_path, key, json.dumps(value), expires_at))

    def search_context_semantic(self, query_text: str, limit: int = 8,
                                max_age_minutes: int = 30) -> list:
        """
        Retrieve the top-N most semantically relevant context rows for a query,
        instead of dumping every category. This is the retrieval mechanism that
        substitutes for weight-level adaptation: the model's inputs are selected
        by relevance, not fixed by category list.
        """
        try:
            from cortex.llm.embeddings import embed, to_pgvector
        except Exception as e:
            logger.warning(f"Embeddings unavailable, falling back to non-semantic context: {e}")
            return []

        query_vec = embed(query_text)
        if not query_vec:
            return []
        vec_literal = to_pgvector(query_vec)

        from datetime import timezone
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=max_age_minutes)

        if self.backend == "sqlite":
            import cortex.storage.sqlite_backend as sqlite_backend
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT category, key, value, collected_at, embedding
                        FROM context_store
                        WHERE embedding IS NOT NULL AND collected_at > %s
                    """, (cutoff,))
                    rows = [dict(r) for r in cur.fetchall()]
            ranked = sqlite_backend.cosine_rank(query_vec, rows, "embedding", limit)
            return [{"category": r["category"], "key": r["key"], "value": r["value"],
                     "collected_at": str(r["collected_at"]), "similarity": float(r["_similarity"])}
                    for r in ranked]

        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT category, key, value, collected_at,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM context_store
                    WHERE embedding IS NOT NULL
                      AND collected_at > %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (vec_literal, cutoff, vec_literal, limit))
                rows = cur.fetchall()

        return [{"category": r["category"], "key": r["key"], "value": r["value"],
                 "collected_at": str(r["collected_at"]), "similarity": float(r["similarity"])}
                for r in rows]

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
                    # IN (...) is dialect-neutral (ANY(%s) is Postgres-only)
                    placeholders = ", ".join(["%s"] * len(categories))
                    query += f" AND category IN ({placeholders})"
                    params.extend(categories)

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

    def get_chat_sessions(self, limit: int = 40) -> list:
        """Recent chat sessions (for the sidebar): session_id + title (first user
        message) + last activity. Real history, newest first."""
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT session_id, MAX(timestamp) AS last_ts, COUNT(*) AS n
                    FROM conversations
                    GROUP BY session_id
                    ORDER BY last_ts DESC
                    LIMIT %s
                """, (limit,))
                sessions = [dict(r) for r in cur.fetchall()]
                out = []
                for s in sessions:
                    cur.execute("""
                        SELECT content FROM conversations
                        WHERE session_id = %s AND role = 'user'
                        ORDER BY id ASC LIMIT 1
                    """, (s["session_id"],))
                    row = cur.fetchone()
                    title = (dict(row)["content"] if row else "") or "New chat"
                    out.append({"session_id": s["session_id"],
                                "title": title.strip()[:60],
                                "last_ts": str(s["last_ts"])})
        return out

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

    def delete_reminder(self, reminder_id: int) -> bool:
        """Delete a reminder by ID. Returns True if deleted."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM reminders WHERE id = %s", (reminder_id,))
                return cur.rowcount > 0

    def update_reminder(self, reminder_id: int, text: str = None,
                        due_at=None, repeat_rule: str = None) -> bool:
        """Update a reminder's text, due time, or repeat rule."""
        updates = []
        params = []
        if text is not None:
            updates.append("text = %s")
            params.append(text)
        if due_at is not None:
            updates.append("due_at = %s")
            params.append(due_at)
        if repeat_rule is not None:
            updates.append("repeat_rule = %s")
            params.append(repeat_rule if repeat_rule != "none" else None)
        if not updates:
            return False
        params.append(reminder_id)
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"UPDATE reminders SET {', '.join(updates)} WHERE id = %s", params)
                return cur.rowcount > 0

    def get_all_reminders(self, include_fired: bool = False) -> list:
        """Get all reminders, optionally including fired ones."""
        import psycopg2.extras
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if include_fired:
                    cur.execute("SELECT * FROM reminders ORDER BY due_at ASC")
                else:
                    cur.execute("SELECT * FROM reminders WHERE fired = FALSE ORDER BY due_at ASC")
                return [dict(r) for r in cur.fetchall()]

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

    # ─── Experiences (State-Action-Reward buffer) ───

    def log_experience(self, session_id: str, user_text: str, action: str,
                       action_payload: dict, speak: str = None,
                       state_context: dict = None) -> int:
        """Log an agent turn as an experience. Reward is computed later, async,
        once the outcome (exit code, correction, feedback) is known."""
        vec_literal = None
        try:
            from cortex.llm.embeddings import embed, to_pgvector
            vec_literal = to_pgvector(embed(user_text))
        except Exception as e:
            logger.warning(f"Experience embedding failed: {e}")

        vec_cast = "::vector" if self.backend == "postgres" else ""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                if vec_literal:
                    cur.execute(f"""
                        INSERT INTO experiences
                            (session_id, user_text, action, action_payload, speak, state_context, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s{vec_cast})
                        RETURNING id
                    """, (session_id, user_text, action, json.dumps(action_payload),
                          speak, json.dumps(state_context) if state_context else None, vec_literal))
                else:
                    cur.execute("""
                        INSERT INTO experiences
                            (session_id, user_text, action, action_payload, speak, state_context)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (session_id, user_text, action, json.dumps(action_payload),
                          speak, json.dumps(state_context) if state_context else None))
                return cur.fetchone()[0]

    def update_experience_outcome(self, experience_id: int, exit_code: int = None,
                                  was_corrected: bool = None, was_confirmed: bool = None,
                                  explicit_feedback: int = None):
        """
        Attach outcome signals to an experience after the fact.
        If reward was already computed by an earlier heuristic pass, force it back
        to NULL so compute_pending_rewards() re-scores it with the new signal.
        Without this, late-arriving signals (e.g. explicit feedback given after
        an exit_code already triggered scoring) would be silently ignored, since
        compute_pending_rewards() only looks at rows WHERE reward IS NULL.
        """
        updates, params = [], []
        if exit_code is not None:
            updates.append("exit_code = %s"); params.append(exit_code)
        if was_corrected is not None:
            updates.append("was_corrected = %s"); params.append(was_corrected)
        if was_confirmed is not None:
            updates.append("was_confirmed = %s"); params.append(was_confirmed)
        if explicit_feedback is not None:
            updates.append("explicit_feedback = %s"); params.append(explicit_feedback)
        if not updates:
            return
        # Any new outcome signal invalidates a previously computed reward
        updates.append("reward = NULL")
        updates.append("reward_computed_at = NULL")
        params.append(experience_id)
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"UPDATE experiences SET {', '.join(updates)} WHERE id = %s", params)

    def compute_pending_rewards(self, batch_size: int = 100) -> int:
        """
        Compute reward for experiences that have outcome signals but no reward yet.
        Heuristic reward function (no RL training loop needed — this is the scoring
        step that makes retrieval reward-aware):
          exit_code == 0            -> +0.5
          exit_code != 0 (attempted)-> -0.5
          was_corrected             -> -0.7   (user had to re-ask / rephrase)
          was_confirmed == False    -> -0.3   (user rejected a proposed action)
          was_confirmed == True     -> +0.2
          explicit_feedback         -> that value directly (-1 / 0 / +1), dominant signal
        Rewards are summed and clamped to [-1, 1].
        """
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, exit_code, was_corrected, was_confirmed, explicit_feedback
                    FROM experiences
                    WHERE reward IS NULL
                      AND (exit_code IS NOT NULL OR was_corrected IS NOT NULL
                           OR was_confirmed IS NOT NULL OR explicit_feedback IS NOT NULL)
                    LIMIT %s
                """, (batch_size,))
                rows = cur.fetchall()

            updated = 0
            with conn.cursor() as cur:
                for r in rows:
                    score = 0.0
                    reasons = []

                    if r["exit_code"] is not None:
                        if r["exit_code"] == 0:
                            score += 0.5; reasons.append("exit_code=0 (+0.5)")
                        else:
                            score -= 0.5; reasons.append(f"exit_code={r['exit_code']} (-0.5)")

                    if r["was_corrected"]:
                        score -= 0.7; reasons.append("user_corrected (-0.7)")

                    if r["was_confirmed"] is not None:
                        if r["was_confirmed"]:
                            score += 0.2; reasons.append("confirmed (+0.2)")
                        else:
                            score -= 0.3; reasons.append("rejected (-0.3)")

                    if r["explicit_feedback"] is not None:
                        score = float(r["explicit_feedback"])
                        reasons = [f"explicit_feedback={r['explicit_feedback']} (overrides)"]

                    score = max(-1.0, min(1.0, score))
                    cur.execute("""
                        UPDATE experiences
                        SET reward = %s, reward_reason = %s, reward_computed_at = NOW()
                        WHERE id = %s
                    """, (score, "; ".join(reasons), r["id"]))
                    updated += 1

        if updated:
            logger.info(f"Computed rewards for {updated} experiences")
        return updated

    def get_similar_experiences(self, query_text: str, limit: int = 5,
                                min_reward: float = 0.0) -> list:
        """
        Retrieve past high-reward experiences similar to the current query.
        This is the practical substitute for BIG BRAIN's fast-weight adaptation:
        instead of updating model weights on this interaction, we retrieve what
        worked before in similar situations and feed it back in as few-shot context.
        """
        try:
            from cortex.llm.embeddings import embed, to_pgvector
        except Exception as e:
            logger.warning(f"Embeddings unavailable for experience retrieval: {e}")
            return []

        query_vec = embed(query_text)
        if not query_vec:
            return []
        vec_literal = to_pgvector(query_vec)

        if self.backend == "sqlite":
            import cortex.storage.sqlite_backend as sqlite_backend
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT user_text, action, action_payload, speak, reward, embedding
                        FROM experiences
                        WHERE embedding IS NOT NULL AND reward IS NOT NULL AND reward >= %s
                    """, (min_reward,))
                    rows = [dict(r) for r in cur.fetchall()]
            ranked = sqlite_backend.cosine_rank(query_vec, rows, "embedding", limit)
            return [{"user_text": r["user_text"], "action": r["action"],
                     "action_payload": r["action_payload"], "speak": r["speak"],
                     "reward": float(r["reward"]), "similarity": float(r["_similarity"])}
                    for r in ranked]

        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT user_text, action, action_payload, speak, reward,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM experiences
                    WHERE embedding IS NOT NULL
                      AND reward IS NOT NULL
                      AND reward >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (vec_literal, min_reward, vec_literal, limit))
                rows = cur.fetchall()

        return [{"user_text": r["user_text"], "action": r["action"],
                 "action_payload": r["action_payload"], "speak": r["speak"],
                 "reward": float(r["reward"]), "similarity": float(r["similarity"])}
                for r in rows]

    def find_recent_experience(self, session_id: str, user_text: str, within_last: int = 1):
        """Find the most recent experience for a session (used to attach outcomes)."""
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id FROM experiences
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (session_id, within_last))
                rows = cur.fetchall()
        return rows[0]["id"] if rows else None

    # ─── Goals + Autonomy Log ───

    def create_goal(self, description: str, priority: int = 3, source: str = "self") -> int:
        """Create a new goal (self-generated or user-seeded). Returns its ID."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO goals (description, priority, source)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (description, priority, source))
                return cur.fetchone()[0]

    def get_active_goals(self, limit: int = 10) -> list:
        """Get active goals, highest priority first."""
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, description, priority, source, created_at, last_evaluated_at
                    FROM goals
                    WHERE status = 'active'
                    ORDER BY priority DESC, created_at ASC
                    LIMIT %s
                """, (limit,))
                return [dict(r) for r in cur.fetchall()]

    def update_goal(self, goal_id: int, status: str = None, result: str = None,
                    touch_evaluated: bool = False):
        """Update a goal's status/result, or just mark it evaluated this cycle."""
        updates, params = [], []
        if status is not None:
            updates.append("status = %s")
            params.append(status)
            if status in ("done", "abandoned"):
                updates.append("completed_at = NOW()")
        if result is not None:
            updates.append("result = %s")
            params.append(result)
        if touch_evaluated:
            updates.append("last_evaluated_at = NOW()")
        if not updates:
            return
        params.append(goal_id)
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"UPDATE goals SET {', '.join(updates)} WHERE id = %s", params)

    def log_autonomy_cycle(self, think: str, decision: str, goal_id: int = None,
                           action_payload: dict = None, needs_confirmation: bool = False,
                           exit_code: int = None) -> int:
        """Record one autonomy loop cycle's decision. Returns its ID."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO autonomy_log
                        (think, decision, goal_id, action_payload, needs_confirmation, exit_code)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (think, decision, goal_id,
                      json.dumps(action_payload) if action_payload else None,
                      needs_confirmation, exit_code))
                return cur.fetchone()[0]

    def get_autonomy_log(self, limit: int = 20) -> list:
        """Most recent autonomy cycles, newest first."""
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, cycle_at, think, decision, goal_id, action_payload,
                           needs_confirmation, confirmed, exit_code, notified
                    FROM autonomy_log
                    ORDER BY cycle_at DESC
                    LIMIT %s
                """, (limit,))
                return [dict(r) for r in cur.fetchall()]

    def count_recent_autonomy_actions(self, hours: float = 1.0) -> int:
        """Count acted/proposed autonomy decisions in the last N hours (rate limiting)."""
        from datetime import timezone
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM autonomy_log
                    WHERE decision IN ('acted', 'proposed')
                      AND cycle_at > %s
                """, (cutoff,))
                return cur.fetchone()[0]

    def get_pending_proposal(self, proposal_id: int):
        """Get a single autonomy proposal awaiting user confirmation."""
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, think, action_payload, goal_id
                    FROM autonomy_log
                    WHERE id = %s AND decision = 'proposed' AND confirmed IS NULL
                """, (proposal_id,))
                return cur.fetchone()

    def resolve_proposal(self, proposal_id: int, confirmed: bool, exit_code: int = None):
        """Mark an autonomy proposal confirmed or denied by the user."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE autonomy_log
                    SET confirmed = %s, exit_code = %s
                    WHERE id = %s
                """, (confirmed, exit_code, proposal_id))

    # ─── Priorities (standing values) ───

    def create_priority(self, statement: str, weight: int = 3, source: str = "user") -> int:
        """Add a standing priority. Returns its ID."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO priorities (statement, weight, source)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (statement, weight, source))
                return cur.fetchone()[0]

    def get_active_priorities(self, limit: int = 20) -> list:
        """Active priorities, highest weight first."""
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, statement, weight, source, created_at,
                           last_reinforced_at, reinforcement_count
                    FROM priorities
                    WHERE status = 'active'
                    ORDER BY weight DESC, created_at ASC
                    LIMIT %s
                """, (limit,))
                return [dict(r) for r in cur.fetchall()]

    def retire_priority(self, priority_id: int) -> bool:
        """Retire a priority (soft delete — keeps history)."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE priorities SET status = 'retired' WHERE id = %s", (priority_id,))
                return cur.rowcount > 0

    def reinforce_priority(self, priority_id: int):
        """Bump a priority's reinforcement count when something re-confirms it matters."""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE priorities
                    SET reinforcement_count = reinforcement_count + 1, last_reinforced_at = NOW()
                    WHERE id = %s
                """, (priority_id,))

    def get_recurring_self_goals(self, min_count: int = 2, days: int = 14) -> list:
        """
        Self-formed goals that keep recurring (near-duplicate description, case/
        whitespace-insensitive) in the last N days — candidates for promotion
        into a standing priority. Simple exact-match-after-normalize grouping,
        no fuzzy matching needed for this scale.
        """
        from datetime import timezone
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT LOWER(TRIM(description)) AS normalized,
                           MAX(description) AS description,
                           COUNT(*) AS occurrences,
                           MAX(created_at) AS last_seen
                    FROM goals
                    WHERE source = 'self' AND created_at > %s
                    GROUP BY LOWER(TRIM(description))
                    HAVING COUNT(*) >= %s
                    ORDER BY occurrences DESC
                """, (cutoff, min_count))
                return [dict(r) for r in cur.fetchall()]

    # ─── RAG documents (uploaded files, chunked + embedded) ───

    def add_document(self, source: str, chunk: str, vec_literal: str = None):
        vec_cast = "::vector" if self.backend == "postgres" else ""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                if vec_literal:
                    cur.execute(f"INSERT INTO documents (source, chunk, embedding) VALUES (%s, %s, %s{vec_cast})",
                                (source, chunk, vec_literal))
                else:
                    cur.execute("INSERT INTO documents (source, chunk) VALUES (%s, %s)", (source, chunk))

    def search_documents(self, query_vec, limit: int = 6) -> list:
        """Top-k most relevant uploaded-doc chunks for a query embedding."""
        if self.backend == "sqlite":
            import cortex.storage.sqlite_backend as sqlite_backend
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT source, chunk, embedding FROM documents WHERE embedding IS NOT NULL")
                    rows = [dict(r) for r in cur.fetchall()]
            ranked = sqlite_backend.cosine_rank(query_vec, rows, "embedding", limit)
            return [{"source": r["source"], "chunk": r["chunk"], "similarity": float(r["_similarity"])}
                    for r in ranked]
        from cortex.llm.embeddings import to_pgvector
        vec = to_pgvector(query_vec)
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("""SELECT source, chunk, 1-(embedding <=> %s::vector) AS similarity
                    FROM documents WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector LIMIT %s""", (vec, vec, limit))
                return [dict(r) for r in cur.fetchall()]

    def list_documents(self) -> list:
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute("SELECT source, COUNT(*) AS chunks FROM documents GROUP BY source ORDER BY MAX(id) DESC")
                return [dict(r) for r in cur.fetchall()]

    def clear_documents(self, source: str = None):
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                if source:
                    cur.execute("DELETE FROM documents WHERE source = %s", (source,))
                else:
                    cur.execute("DELETE FROM documents")

    # ─── Nudges (proactive notifications Cortex surfaces to the user) ───

    def add_nudge(self, title: str, body: str = "", kind: str = "info",
                  source: str = "autonomy") -> int:
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO nudges (title, body, kind, source) VALUES (%s, %s, %s, %s) RETURNING id",
                    (title, body, kind, source))
                return cur.fetchone()[0]

    def get_nudges(self, unread_only: bool = False, limit: int = 20) -> list:
        q = "SELECT id, title, body, kind, source, read, created_at FROM nudges"
        if unread_only:
            q += " WHERE read = 0"
        q += " ORDER BY id DESC LIMIT %s"
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(q, (limit,))
                return [dict(r) for r in cur.fetchall()]

    def count_unread_nudges(self) -> int:
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM nudges WHERE read = 0")
                return int(cur.fetchone()[0] or 0)

    def mark_nudges_read(self, nudge_id: int = None):
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                if nudge_id:
                    cur.execute("UPDATE nudges SET read = 1 WHERE id = %s", (nudge_id,))
                else:
                    cur.execute("UPDATE nudges SET read = 1 WHERE read = 0")

    def clear_nudges(self):
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM nudges")

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
