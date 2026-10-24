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
    """
    Return a connected dual-backend Database (local-first). Collectors used to
    open a raw psycopg2 connection to the remote Postgres — which broke in the
    local-first model (their data never reached Cortex's local SQLite, and every
    run failed when Tailscale was off). Now they write wherever Cortex reads.
    """
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
    from cortex.storage.db import Database
    dbc = config["database"]
    db = Database(
        backend=dbc.get("backend", "sqlite"),
        sqlite_path=os.path.expanduser(dbc.get("sqlite_path", "~/.cortex/cortex.db")),
        host=dbc.get("host"), port=dbc.get("port"), name=dbc.get("name"),
        user=dbc.get("user"),
        password=dbc.get("password") or os.environ.get("CORTEX_DB_PASSWORD", ""),
    )
    if not db.connect():
        raise RuntimeError("collector: DB connection failed")
    return db


def upsert_context(conn, category: str, key: str, value: dict,
                   project_path: str = None, expires_minutes: int = 15,
                   embedding_text: str = None):
    """Write a context row (insert or update) through the dual-backend Database."""
    conn.upsert_context(category, key, value, project_path=project_path,
                        expires_minutes=expires_minutes, embedding_text=embedding_text)


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
    """Fetch all tracked projects via the dual-backend Database."""
    return [dict(r) for r in conn.get_tracked_projects()]
