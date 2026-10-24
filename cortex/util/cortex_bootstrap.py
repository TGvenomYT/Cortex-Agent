"""
cortex_bootstrap.py — minimal dependency construction for scheduled jobs.

The old model built everything at import time inside cortex.py (the always-on
Flask daemon). In the scheduled (wake-run-exit) model each launchd job only
needs a subset of components and must NOT start any background threads or an
HTTP server. This module builds exactly what a one-shot job needs and returns
it, reusing the same classes and config loading as the daemon.

Nothing here starts a thread or binds a socket.
"""
from __future__ import annotations

import os
import sys
import subprocess
import logging

import yaml

logger = logging.getLogger("cortex.bootstrap")

# macOS Keychain service name for Cortex secrets (encrypted at rest).
_KEYCHAIN_SERVICE = "cortex"


def keychain_get(account: str) -> str | None:
    """Read a secret from the OS secret store (macOS Keychain / Windows
    Credential Manager / Linux Secret Service, with a 0600-file fallback)."""
    import cortex.util.osutil as osutil
    return osutil.secret_get(account)


def keychain_set(account: str, secret: str) -> bool:
    """Store/replace a secret in the OS secret store (cross-platform)."""
    import cortex.util.osutil as osutil
    return osutil.secret_set(account, secret)


def load_secrets() -> None:
    """
    Populate secrets into os.environ, preferring the most secure source:
    1. an already-set environment variable (explicit override),
    2. the macOS Keychain (encrypted at rest — the 'ciphertext' store),
    3. the 0600 ~/.cortex/cortex.env file (fallback).
    """
    if not os.environ.get("CORTEX_DB_PASSWORD"):
        pw = keychain_get("cortex-db-password")
        if pw:
            os.environ["CORTEX_DB_PASSWORD"] = pw
    load_env_file()

CONFIG_PATHS = [
    os.path.join(__import__("cortex").ROOT, "config.yaml"),
    os.path.expanduser("~/.cortex/config.yaml"),
]

# Where per-run state (cooldowns, cycle counters) lives in the scheduled model,
# so it survives across wake-run-exit invocations. Kept out of the DB so a
# flaky Tailscale/Supabase link doesn't wipe local scheduling state.
STATE_DIR = os.path.expanduser("~/.cortex/state")


ENV_FILE = os.path.expanduser("~/.cortex/cortex.env")


def ensure_state_dir() -> str:
    os.makedirs(STATE_DIR, mode=0o700, exist_ok=True)
    return STATE_DIR


def load_env_file(path: str = ENV_FILE) -> None:
    """
    Populate os.environ from a 0600 KEY=VALUE env file (the DB password lives
    here, not in the launchd plists). Values already present in the environment
    win, so an explicit CORTEX_DB_PASSWORD still overrides. This lets launchd
    invoke the venv python directly — no shell wrapper, which macOS TCC blocks
    from running inside ~/Documents.
    """
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())
    except FileNotFoundError:
        pass


def load_config() -> dict:
    for path in CONFIG_PATHS:
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)
    logger.error("No config.yaml found")
    sys.exit(1)


def connect_db(config: dict):
    """
    Connect to the Cortex DB. Local-first: defaults to SQLite (works offline),
    falls back to the Postgres config only if backend is explicitly 'postgres'.
    Returns a connected Database or None.
    """
    from cortex.storage.db import Database
    dbc = config["database"]
    backend = dbc.get("backend", "sqlite")
    db = Database(
        backend=backend,
        sqlite_path=os.path.expanduser(dbc.get("sqlite_path", "~/.cortex/cortex.db")),
        host=dbc.get("host"),
        port=dbc.get("port"),
        name=dbc.get("name"),
        user=dbc.get("user"),
        password=dbc.get("password") or os.environ.get("CORTEX_DB_PASSWORD", ""),
    )
    if not db.connect():
        logger.error("DB connection failed")
        return None
    return db


def build_llm(config: dict):
    from cortex.llm.llm import create_router
    return create_router(config["llm"])


def build_for_autonomy(config: dict):
    """
    Everything the autonomy session needs: db, llm, executor, context, memory.
    Returns a dict, or None if the DB is unreachable (job should skip cleanly).
    """
    db = connect_db(config)
    if db is None:
        return None
    from cortex.core.executor import CommandExecutor
    from cortex.core.context_reader import ContextReader
    from cortex.core.memory import MemoryManager
    llm = build_llm(config)
    return {
        "db": db,
        "llm": llm,
        "executor": CommandExecutor(db),
        "context_reader": ContextReader(db, config.get("context", {})),
        "memory_mgr": MemoryManager(db, llm, config.get("memory", {})),
        "config": config,
    }
