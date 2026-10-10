"""
Collector base — shared config loader, DB connection, upsert helper.
All collectors import from here. Keeps each collector lean (~50 lines of logic).
"""

import os
import sys
import subprocess
import logging

import yaml
import psycopg2
import psycopg2.extras

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

# ─── Config ───
CONFIG_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "config.yaml"),
    os.path.expanduser("~/.cortex/config.yaml"),
]


def load_config() -> dict:
    for path in CONFIG_PATHS:
        path = os.path.normpath(path)
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)
    print("ERROR: No config.yaml found", file=sys.stderr)
    sys.exit(1)


def get_db_conn(config: dict):
    """Return a raw psycopg2 connection (autocommit on)."""
    db = config["database"]
    password = db.get("password") or os.environ.get("CORTEX_DB_PASSWORD", "")
    conn = psycopg2.connect(
        host=db["host"],
        port=db["port"],
        dbname=db["name"],
        user=db["user"],
        password=password,
        connect_timeout=10
    )
    conn.autocommit = True
    return conn


def upsert_context(conn, category: str, key: str, value: dict,
                   project_path: str = None, expires_minutes: int = 15,
                   embedding_text: str = None):
    """
    Write a context row to the DB (insert or update).
    embedding_text: optional text to embed for semantic retrieval (see embeddings.py).
    If provided, this row becomes findable via db.search_context_semantic().
    """
    import json
    from datetime import datetime, timedelta
    expires_at = datetime.utcnow() + timedelta(minutes=expires_minutes)

    vec_literal = None
    if embedding_text:
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from embeddings import embed, to_pgvector
            vec_literal = to_pgvector(embed(embedding_text))
        except Exception as e:
            logging.getLogger("collector.base").warning(f"Embedding failed for {category}/{key}: {e}")

    with conn.cursor() as cur:
        if vec_literal:
            cur.execute("""
                INSERT INTO context_store (category, project_path, key, value, collected_at, expires_at, embedding)
                VALUES (%s, %s, %s, %s, NOW(), %s, %s::vector)
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


def run_cmd(cmd: str, cwd: str = None, timeout: int = 10) -> tuple[str, str, int]:
    """Run a shell command, return (stdout, stderr, returncode)."""
    # Ensure PATH includes common tool locations (launchd has minimal PATH)
    env = os.environ.copy()
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:" + env.get("PATH", "")
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=timeout,
            env=env
        )
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", f"timeout after {timeout}s", -1
    except Exception as e:
        return "", str(e), -1


def get_tracked_projects(conn) -> list[dict]:
    """Fetch all tracked projects from DB."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT name, path, type, custom_commands FROM tracked_projects")
        return [dict(r) for r in cur.fetchall()]
